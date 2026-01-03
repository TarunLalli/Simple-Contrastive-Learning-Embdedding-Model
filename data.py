from datasets import load_dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch

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

#Inspecting data
#for line in train_text:
#    if line.strip():   # non-empty
#        print(line)
#        break

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

#Inspecting vocabulary
#Vocab length
#print(f"Vocab Length:{vocab.__len__()}")
#Checking <unk>
#print(f"Index for <unk>: {vocab.__getitem__('<unk>')}")
#Verifying 2/3 examples
#for token in ['=', 'valkyria', 'chronicles']:
    #print(f"Index for {token}: {vocab.__getitem__(token)}")
#Verifying unseen example
#print(f"Index for Fake example: {vocab.__getitem__('dskjckdjcbskjc')}")

#Forming training corpus
training_corpus = []
for line in train_text:
    for token in tokenizer(line):
        training_corpus.append(vocab.__getitem__(token))
training_corpus = torch.tensor(training_corpus)