import numpy as np
import torch
from transformers import AutoModelForTokenClassification


class GbcNlpService:
    DEVICE_TYPE = 'cuda'
    DATA_DIR = 'preprocess/data/'
    DATA_SET = '.csv'
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

    def _accuracy_per_class(self, preds, labels):
        label_dict_inverse = {v: k for k, v in self.label_dict.items()}

        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        for label in np.unique(labels_flat):
            y_preds = preds_flat[labels_flat == label]
            y_true = labels_flat[labels_flat == label]
            print(f'Class: {label_dict_inverse[label]}')
            print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}'
                  f' -> {len(y_preds[y_preds == label]) / len(y_true)}\n')

    def _evaluate(self):
        self.model.eval()

        # label_list = [
        #     "O",  # Outside of a named entity
        #     "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
        #     "I-MISC",  # Miscellaneous entity
        #     "B-PER",  # Beginning of a person's name right after another person's name
        #     "I-PER",  # Person's name
        #     "B-ORG",  # Beginning of an organisation right after another organisation
        #     "I-ORG",  # Organisation
        #     "B-LOC",  # Beginning of a location right after another location
        #     "I-LOC"  # Location
        # ]
        #
        # sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO"
        #
        # # Bit of a hack to get the tokens with the special tokens
        # tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
        # inputs = tokenizer.encode(sequence, return_tensors="pt")
        #
        # outputs = model(inputs)[0]
        # predictions = torch.argmax(outputs, dim=2)
        #
        # print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())])

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

    def predict(self):
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.BERT_MODEL,
            num_labels=len(self.label_dict),
            output_attentions=False,
            output_hidden_states=False)

        self.model.to(self.device)

        self.model.load_state_dict(
            torch.load('data_volume/finetuned_BERT_epoch_5.model',
                       map_location=torch.device(self.DEVICE_TYPE)))

        _, predictions, true_vals = self._evaluate()
        self._accuracy_per_class(predictions, true_vals)


if __name__ == '__main__':
    text_service = GbcNlpService()

    print(f"\nPrediction...")
    text_service.predict()
