import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

id2label = {
    "0": "B-LOC",
    "1": "B-MISC",
    "2": "B-ORG",
    "3": "B-PER",
    "4": "I-LOC",
    "5": "I-MISC",
    "6": "I-ORG",
    "7": "I-PER",
    "8": "O"
}

tokenizer = AutoTokenizer.from_pretrained('mrm8488/RuPERTa-base-finetuned-ner')
model = AutoModelForTokenClassification.from_pretrained('mrm8488/RuPERTa-base-finetuned-ner')

text = r"Julien, CEO de HF, nació en Francia."
input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)

outputs = model(input_ids)
last_hidden_states = outputs[0]

for m in last_hidden_states:
    for index, n in enumerate(m):
        if 0 < index <= len(text.split(" ")):
            print(text.split(" ")[index - 1] + ": " + id2label[str(torch.argmax(n).item())])
