from pprint import pprint

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
encoded_input = tokenizer("Hola, esto es una simple sentencia!")
pprint(tokenizer.decode(encoded_input["input_ids"]))
pprint(encoded_input)

device = torch.device("cuda")
df = pd.read_csv("data/spamdata_v2.csv")
df.head()

# split train dataset into train, validation and test sets
train_text, temp_text, train_labels, temp_labels = train_test_split(df['text'], df['label'],
                                                                    random_state=2018,
                                                                    test_size=0.3,
                                                                    stratify=df['label'])

val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=2018,
                                                                test_size=0.5,
                                                                stratify=temp_labels)

seq_len = [len(i.split()) for i in train_text]
pd.Series(seq_len).hist(bins=30)
