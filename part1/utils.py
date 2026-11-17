import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import re
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

TYPO_PROB = 0.30
_NEIGHBOURS = {
    "q": "wa",
    "w": "qes",
    "e": "wrd",
    "r": "etf",
    "t": "ryg",
    "y": "tuh",
    "u": "yij",
    "i": "uok",
    "o": "ipl",
    "p": "ol",
    "a": "qsz",
    "s": "awdx",
    "d": "sefc",
    "f": "drgv",
    "g": "fthb",
    "h": "gyjn",
    "j": "hukm",
    "k": "jiol",
    "l": "kop",
    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
}


def _add_typo(word):
    positions = [i for i, ch in enumerate(word) if ch.isalpha()]
    if len(positions) < 2:
        return word
    i = random.choice(positions[1:])
    ch = word[i]
    neighbours = _NEIGHBOURS.get(ch.lower())
    if not neighbours:
        return word
    new = random.choice(neighbours)
    if ch.isupper():
        new = new.upper()
    return word[:i] + new + word[i + 1 :]


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # Replace the words of the review by keyboard-neighbour typos
    def typo(match):
        word = match.group(0)
        if len(word) >= 3 and random.random() < TYPO_PROB:
            return _add_typo(word)
        return word

    # [A-Za-z]+ matches whole words and leaves punctuation/spacing untouched
    example["text"] = re.sub(r"[A-Za-z]+", typo, example["text"])

    ##### YOUR CODE ENDS HERE ######

    return example
