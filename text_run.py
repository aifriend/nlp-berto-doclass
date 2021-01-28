import os
import random
from pprint import pprint

import numpy as np
import torch
from pandas import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
from transformers import BertTokenizer


class GbcNlpService:
    DEVICE_TYPE = 'cuda'
    DATA_DIR = 'data/from_text'
    DATA_SET = 'title_conference.csv'
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 3
    NUM_EPOCHS = 5

    # BETO Bert spanish pre-trained
    # 'dccuchile/bert-base-spanish-wwm-uncased'
    BERT_MODEL = 'bert-base-uncased'

    def __init__(self):
        self.epochs = self.NUM_EPOCHS

        self.df = None
        self.label_dict = {}
        self.dataset_train = None
        self.dataset_val = None
        self.model = None
        self.data_loader_train = None
        self.data_loader_validation = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device(
            self.DEVICE_TYPE if torch.cuda.is_available() else "cpu")

    def load_data(self):
        # read dataset from CSV
        self.df = pd.read_csv(os.path.join(self.DATA_DIR, self.DATA_SET))
        self.df.head()
        print(f"\n{self.df['Category'].value_counts()}")

        possible_labels = self.df.Category.unique()
        self.label_dict = {}
        for index, possible_label in enumerate(possible_labels):
            self.label_dict[possible_label] = index
        print(f"\n{self.label_dict}")

        # add label_text by numbers
        self.df['label'] = self.df.Category.replace(self.label_dict)

        """
        Because the labels are imbalanced,
        we split the data set in a stratified fashion,
        using this as the class labels.
        """
        X_train, X_val, y_train, y_val = train_test_split(self.df.index.values,
                                                          self.df.label.values,
                                                          test_size=0.15,
                                                          random_state=42,
                                                          stratify=self.df.label.values)

        self.df['data_type'] = ['not_set'] * self.df.shape[0]
        self.df.loc[X_train, 'data_type'] = 'train'
        self.df.loc[X_val, 'data_type'] = 'val'
        print(f"\n{self.df.groupby(['Category', 'label', 'data_type']).count()}")

    def tokenizer(self):
        """
        Tokenization is a process to take raw texts and split into tokens, which are numeric data to represent words.
        Constructs a BERT tokenizer. Based on WordPiece.
        Instantiate a pre-trained BERT model configuration to encode our data.
        To convert all the titles from text into encoded form, we use a function called batch_encode_plus,
        We will proceed train and validation data separately.
        The 1st parameter inside the above function is the title text.
        add_special_tokens=True means the sequences will be encoded with the special tokens relative to their model.
        When batching sequences together, we set return_attention_mask=True,
        so it will return the attention mask according to the specific tokenizer defined by the max_length attribute.
        We also want to pad all the titles to certain maximum length.
        We actually do not need to set max_length=256, but just to play it safe.
        return_tensors='pt' to return PyTorch.
        And then we need to split the data into input_ids, attention_masks and labels.
        Finally, after we get encoded data set, we can create training data and validation data.
        """
        tokenizer = BertTokenizer.from_pretrained(
            self.BERT_MODEL,
            use_fast=False,
            strip_accents=True,
            do_lower_case=True)

        encoded_data_train = tokenizer.batch_encode_plus(
            self.df[self.df.data_type == 'train'].Title.values,
            add_special_tokens=True,
            return_attention_mask=True,
            padding='longest',
            return_tensors='pt'
        )
        pprint(encoded_data_train)
        # seq_len = [len(i.split()) for i in self.df.Title.values]
        # pd.Series(seq_len).plot.hist(bins=30)
        # train_text.tolist(),
        # max_length = 25,
        # pad_to_max_length = True,
        # truncation = True

        encoded_data_val = tokenizer.batch_encode_plus(
            self.df[self.df.data_type == 'val'].Title.values,
            add_special_tokens=True,
            return_attention_mask=True,
            padding='longest',
            return_tensors='pt'
        )
        pprint(encoded_data_val)

        input_ids_train = encoded_data_train['input_ids']
        attention_masks_train = encoded_data_train['attention_mask']
        labels_train = torch.tensor(self.df[self.df.data_type == 'train'].label.values)
        self.dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)

        input_ids_val = encoded_data_val['input_ids']
        attention_masks_val = encoded_data_val['attention_mask']
        labels_val = torch.tensor(self.df[self.df.data_type == 'val'].label.values)
        self.dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

        # Data Loaders
        self.data_loader_train = DataLoader(self.dataset_train,
                                            sampler=RandomSampler(self.dataset_train),
                                            batch_size=self.BATCH_SIZE)

        self.data_loader_validation = DataLoader(self.dataset_val,
                                                 sampler=SequentialSampler(self.dataset_val),
                                                 batch_size=self.BATCH_SIZE)

    def modeling(self):
        """
        We are treating each text as its unique sequence, so one sequence will be classified to one labels
        "model/beto_pytorch_uncased" is a smaller pre-trained model.
        Using num_labels to indicate the number of output labels.
        We don’t really care about output_attentions.
        We also don’t need output_hidden_states.
        DataLoader combines a dataset and a sampler, and provides an iterable over the given dataset.
        We use RandomSampler for training and SequentialSampler for validation.
        Given the limited memory in my environment, I set batch_size=3.
        """
        self.model = BertForSequenceClassification.from_pretrained(
            self.BERT_MODEL,
            num_labels=len(self.label_dict),
            output_attentions=False,
            output_hidden_states=False)

        self.model.to(self.DEVICE_TYPE)

        """
        To construct an optimizer, we have to give it an iterable containing the parameters to optimize.
        Then, we can specify optimizer-specific options such as the learning rate, epsilon, etc.
        I found epochs=5 works well for this data set.
        Create a schedule with a learning rate that decreases linearly from the initial learning rate
            set in the optimizer to 0, after a warmup period during which it increases linearly from 0 to
            the initial learning rate set in the optimizer.
        """
        self.optimizer = AdamW(self.model.parameters(),
                               lr=self.LEARNING_RATE,
                               eps=1e-8)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.data_loader_train) * self.epochs)

    @staticmethod
    def _f1_score_func(preds, labels):
        """
        We will use f1 score and accuracy per class as performance metrics.

        :param preds:
        :param labels:
        """
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average='weighted')

    def _accuracy_per_class(self, preds, labels):
        label_dict_inverse = {v: k for k, v in self.label_dict.items()}

        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        for label in np.unique(labels_flat):
            y_preds = preds_flat[labels_flat == label]
            y_true = labels_flat[labels_flat == label]
            print(f'Class: {label_dict_inverse[label]}')
            print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')

    def _evaluate(self):
        self.model.eval()

        loss_val_total = 0
        predictions, true_vals = [], []

        for batch in self.data_loader_validation:
            batch = tuple(b.to(self.device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            with torch.no_grad():
                outputs = self.model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

        loss_val_avg = loss_val_total / len(self.data_loader_validation)

        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)

        return loss_val_avg, predictions, true_vals

    def train(self):
        seed_val = 17
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        for epoch in tqdm(range(1, self.epochs + 1)):
            # set train mode
            self.model.train()

            loss_train_total = 0

            progress_bar = tqdm(
                self.data_loader_train, desc='Epoch {:1d}'.format(epoch),
                leave=False, position=0, disable=False)
            for batch in progress_bar:
                self.model.zero_grad()

                batch = tuple(b.to(self.device) for b in batch)

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[2],
                          }

                outputs = self.model(**inputs)

                loss = outputs[0]
                loss_train_total += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()

                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

            torch.save(self.model.state_dict(), f'data_volume/finetuned_BERT_epoch_{epoch}.model')

            tqdm.write(f'\nEpoch {epoch}')

            loss_train_avg = loss_train_total / len(self.data_loader_train)
            tqdm.write(f'Training loss: {loss_train_avg}')

            val_loss, predictions, true_vals = self._evaluate()
            tqdm.write(f'Validation loss: {val_loss}')

            val_f1 = self._f1_score_func(predictions, true_vals)
            tqdm.write(f'F1 Score (Weighted): {val_f1}')

    def predict(self):
        self.model = BertForSequenceClassification.from_pretrained(
            self.BERT_MODEL,
            num_labels=len(self.label_dict),
            output_attentions=False,
            output_hidden_states=False)

        self.model.to(self.device)

        self.model.load_state_dict(
            torch.load('data_volume/finetuned_BERT_epoch_1.model',
                       map_location=torch.device(self.DEVICE_TYPE)))

        _, predictions, true_vals = self._evaluate()
        self._accuracy_per_class(predictions, true_vals)


if __name__ == '__main__':
    text_service = GbcNlpService()

    print(f"\nPre-process data from {GbcNlpService.DATA_SET}...")
    text_service.load_data()

    print(f"\nTokenizing data...")
    text_service.tokenizer()

    print(f"\nCreate model from {GbcNlpService.BERT_MODEL}...")
    text_service.modeling()

    print(f"\nTraining...")
    text_service.train()

    print(f"\nPrediction...")
    text_service.predict()
