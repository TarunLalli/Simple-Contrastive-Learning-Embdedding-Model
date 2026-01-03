from datasets import load_dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
import random

#Importing dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

#Obtaining train, test, validation splits
train_text = dataset["train"]["text"]
valid_text = dataset["validation"]["text"]
test_text  = dataset["test"]["text"]

#Removing blank lines
train_text = [line for line in train_text if line.strip()]
valid_text = [line for line in valid_text if line.strip()]
test_text  = [line for line in test_text if line.strip()]

#Initialising tokenizer
tokenizer = get_tokenizer("basic_english")

#Inspoecting tokeisation using example instance 0
print(str(f"Raw text example:{train_text[0]}"))
print(f"Tokenized text example:{tokenizer(train_text[0])}")

#Defining iterator
def train_iter():
    for line in train_text:
        yield tokenizer(line)

#Building Vocabulary
vocab = build_vocab_from_iterator(train_iter(), specials=["<unk>"])

#Setting Vocab default
vocab.set_default_index(vocab.__getitem__('<unk>'))

#Token dropout function set to p = 0.2, this value was chosen as it allows for surface level perturbations without impacting semantic consistency
def word_dropout(tokens, p=0.2):
    kept = [tok for tok in tokens if random.random() > p]
    if len(kept) == 0:
        return tokens  # safety: preserve semantics
    return kept


