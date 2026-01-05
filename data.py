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

#Dataset class returns two views for an individual sample.
class Dataset(torch.utils.data.Dataset):
    def __init__(self,p_dropout, data, tokenizer, vocab):
        super().__init__()
        self.p_dropout = p_dropout
        self.data = data
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __getitem__(self,idx):
        #Fetching raw data/text
        data_idx = self.data[idx]

        #Tokenizing
        data_idx = self.tokenizer(data_idx)

        #Applying dropout
        #View 1
        data_view1 = self.word_dropout(data_idx)
        #View 2
        data_view2 = self.word_dropout(data_idx)

        #Vocab lookups
        #View 1
        token_ids_view1 = [self.vocab.__getitem__(token) for token in data_view1]
        #View 2
        token_ids_view2 = [self.vocab.__getitem__(token) for token in data_view2]

        #Returning both views
        return token_ids_view1, token_ids_view2

    def __len__(self):
        return len(self.data)

    def word_dropout(self,tokens):
        kept = [tok for tok in tokens if random.random() > self.p_dropout]
        if len(kept) == 0:
            return tokens  # safety: preserve semantics
        return kept

class DataLoader(torch.utils.data.DataLoader):
    def __init__(self,dataset):
        super().__init__()

    def collate_fn(batch): 
    #INPUT: Expects a batched 2 dimensional tensor of shape (B,2): [(v1₁, v2₁), (v1₂, v2₂), ..., (v1_B, v2_B)]
    #OUTPUT: Expected output will be 2 tensors of shape (B,L) where L is padded length and B is batch size
        views1 = [
            sample[0]
            for sample in batch
        ] #views1 is of format (v1_1, v1_2, v1_3, ..., v1_B)
        views2 = [
            sample[1]
            for sample in batch
        ] #views2 is of format (v2_1, v2_2, v2_3, ..., v2_B)

        L_max = len(max(views1+views2, key=len))


