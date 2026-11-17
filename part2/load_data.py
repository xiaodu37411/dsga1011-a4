import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import T5TokenizerFast

PAD_IDX = 0


def _input_ids(encoded):
    return _as_1d_long_tensor(encoded["input_ids"])


def _as_1d_long_tensor(sequence):
    return torch.as_tensor(sequence, dtype=torch.long).reshape(-1)


class T5Dataset(Dataset):
    """Tokenized text-to-SQL examples for the T5 training and test splits."""

    def __init__(self, data_folder: str, split: str):
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.data_folder = data_folder
        self.split = split
        self.nl_tokenized, self.sql_tokenized = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder: str, split: str, tokenizer: T5TokenizerFast):
        nl_lines = load_lines(os.path.join(data_folder, f"{split}.nl"))
        sql_lines = None if split == "test" else load_lines(os.path.join(data_folder, f"{split}.sql"))

        # Keep only token IDs so the collator can pad directly into batch tensors.
        nl_tokenized = [_input_ids(tokenizer(line, return_tensors="pt")) for line in nl_lines]
        sql_tokenized = None
        if sql_lines is not None:
            sql_tokenized = [_input_ids(tokenizer(line, return_tensors="pt")) for line in sql_lines]
        return nl_tokenized, sql_tokenized

    def __len__(self):
        return len(self.nl_tokenized)

    def __getitem__(self, idx: int):
        if self.split == "test":
            return self.nl_tokenized[idx]
        return self.nl_tokenized[idx], self.sql_tokenized[idx]


def normal_collate_fn(batch):
    """Dynamically pad a training or development batch."""
    encoder_sequences, target_sequences = zip(*batch)
    encoder_sequences = [_as_1d_long_tensor(sequence) for sequence in encoder_sequences]
    target_sequences = [_as_1d_long_tensor(sequence) for sequence in target_sequences]

    encoder_ids = pad_sequence(encoder_sequences, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = encoder_ids.ne(PAD_IDX).long()

    # T5 uses its pad token as the decoder start token. Shift each target right
    # by one position while retaining EOS in decoder_targets.
    decoder_inputs = [torch.cat((target.new_tensor([PAD_IDX]), target[:-1])) for target in target_sequences]
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(target_sequences, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    """Dynamically pad a test batch and return the decoder start token."""
    encoder_sequences = [_as_1d_long_tensor(sequence) for sequence in batch]
    encoder_ids = pad_sequence(encoder_sequences, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = encoder_ids.ne(PAD_IDX).long()
    initial_decoder_inputs = torch.full(
        (len(batch), 1), PAD_IDX, dtype=torch.long
    )  # torch.zeros is also okey, but PAD_IDX is a variant.
    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = "data"
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn
    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, "r") as f:
        return [line.strip() for line in f]


def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x
