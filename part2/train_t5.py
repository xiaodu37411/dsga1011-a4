import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

from t5_utils import (
    initialize_model,
    initialize_optimizer_and_scheduler,
    save_model,
    load_model_from_checkpoint,
    get_model_tag,
)
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PAD_IDX = 0


def get_args():
    """
    Arguments for training. You may choose to change or extend these as you see fit.
    """
    parser = argparse.ArgumentParser(description="T5 training loop")

    # Model hyperparameters
    parser.add_argument("--finetune", action="store_true", help="Whether to finetune T5 or not")

    # Training hyperparameters
    parser.add_argument("--optimizer_type", type=str, default="AdamW", choices=["AdamW"], help="What optimizer to use")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["none", "cosine", "linear"],
        help="Whether to use a LR scheduler and what type to use if so",
    )
    parser.add_argument(
        "--num_warmup_epochs",
        type=int,
        default=1,
        help="How many epochs to warm up the learning rate for if using a scheduler",
    )
    parser.add_argument("--max_n_epochs", type=int, default=30, help="How many epochs to train the model for")
    parser.add_argument(
        "--patience_epochs",
        type=int,
        default=3,
        help="If validation performance stops improving, how many epochs should we wait before stopping?",
    )

    parser.add_argument("--experiment_name", type=str, default="experiment", help="How should we name this experiment?")
    parser.add_argument(
        "--model_tag",
        type=str,
        default=None,
        choices=["ft", "scr"],
        help="Prefix used in filenames; defaults to 'ft' with --finetune, else 'scr'",
    )

    # Data hyperparameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)

    # Generation hyperparameters
    parser.add_argument(
        "--max_length", type=int, default=512, help="Max length of the generated SQL (data max is 511 tokens)"
    )
    parser.add_argument("--num_beams", type=int, default=1, help="1 = greedy decoding")
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=0,
        help="Block repeated n-grams during generation (0 = disabled). Useful for from-scratch models stuck in repetition loops",
    )
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="1.0 = no penalty")

    args = parser.parse_args()
    return args


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = get_model_tag(args)
    checkpoint_dir = os.path.join("checkpoints", f"{model_type}_experiments", args.experiment_name)
    gt_sql_path = os.path.join(f"data/dev.sql")
    gt_record_path = os.path.join(f"records/ground_truth_dev.pkl")
    model_sql_path = os.path.join(f"results/t5_{model_type}_{args.experiment_name}_dev.sql")
    model_record_path = os.path.join(f"records/t5_{model_type}_{args.experiment_name}_dev.pkl")
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
        )
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.bfloat16,
            enabled=(DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()),
        ):
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )["logits"]

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad].float(), decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens


def eval_epoch(
    args,
    model: T5ForConditionalGeneration,
    dev_loader: DataLoader,
    gt_sql_pth: str,
    model_sql_path: str,
    gt_record_path: str,
    model_record_path: str,
):
    """
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate.

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    """
    model.eval()
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    criterion = nn.CrossEntropyLoss()
    decoded_sqls = []
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs in dev_loader:
            encoder_ids = encoder_ids.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_inputs = decoder_inputs.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            initial_decoder_inputs = initial_decoder_inputs.to(DEVICE)
            logits = model(
                input_ids=encoder_ids,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_inputs,
            ).logits
            non_pad = decoder_targets != PAD_IDX
            num_tokens = non_pad.sum().item()
            total_loss += criterion(logits[non_pad].float(), decoder_targets[non_pad]).item() * num_tokens
            total_tokens += num_tokens
            generated = model.generate(
                input_ids=encoder_ids,
                attention_mask=encoder_mask,
                decoder_input_ids=initial_decoder_inputs,
                max_length=args.max_length,
                num_beams=args.num_beams,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
            )
            decoded_sqls += tokenizer.batch_decode(generated, skip_special_tokens=True)

    save_queries_and_records(decoded_sqls, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    error_rate = sum(1 for msg in error_msgs if msg) / len(decoded_sqls)
    return total_loss / total_tokens, record_f1, record_em, sql_em, error_rate


def test_inference(
    args, model: T5ForConditionalGeneration, test_loader: DataLoader, model_sql_path: str, model_record_path: str
):
    """
    You must implement inference to compute your model's generated SQL queries and its associated
    database records. Implementation should be very similar to eval_epoch.
    """
    model.eval()
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    decoded_sqls = []

    with torch.no_grad():
        for encoder_ids, encoder_mask, initial_decoder_inputs in test_loader:
            encoder_ids = encoder_ids.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            initial_decoder_inputs = initial_decoder_inputs.to(DEVICE)
            generated = model.generate(
                input_ids=encoder_ids,
                attention_mask=encoder_mask,
                decoder_input_ids=initial_decoder_inputs,
                max_length=args.max_length,
                num_beams=args.num_beams,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
            )
            decoded_sqls += tokenizer.batch_decode(generated, skip_special_tokens=True)
    save_queries_and_records(decoded_sqls, model_sql_path, model_record_path)


def main():
    # Get key arguments
    args = get_args()

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    args.num_beams = 4
    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    # Dev set
    experiment_name = args.experiment_name
    model_type = get_model_tag(args)
    gt_sql_path = os.path.join(f"data/dev.sql")
    gt_record_path = os.path.join(f"records/ground_truth_dev.pkl")
    model_sql_path = os.path.join(f"results/t5_{model_type}_{experiment_name}_dev.sql")
    model_record_path = os.path.join(f"records/t5_{model_type}_{experiment_name}_dev.pkl")
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    print(
        f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}"
    )
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f"results/t5_{model_type}_{experiment_name}_test.sql")
    model_record_path = os.path.join(f"records/t5_{model_type}_{experiment_name}_test.pkl")
    test_inference(args, model, test_loader, model_sql_path, model_record_path)


if __name__ == "__main__":
    main()
