import json
import logging
import os
from pprint import pprint

from tokenization import FullTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask


def convert_examples_to_features(examples, _label_list, _max_seq_length, _tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(_label_list, 1)}

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid_ids = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = _tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid_ids.append(1)
                    label_mask.append(True)
                else:
                    valid_ids.append(0)
        if len(tokens) >= _max_seq_length - 1:
            tokens = tokens[0:(_max_seq_length - 2)]
            labels = labels[0:(_max_seq_length - 2)]
            valid_ids = valid_ids[0:(_max_seq_length - 2)]
            label_mask = label_mask[0:(_max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid_ids.insert(0, 1)
        label_mask.insert(0, True)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid_ids.append(1)
        label_mask.append(True)
        label_ids.append(label_map["[SEP]"])
        input_ids = _tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [True] * len(label_ids)
        while len(input_ids) < _max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid_ids.append(1)
            label_mask.append(False)
        while len(label_ids) < _max_seq_length:
            label_ids.append(0)
            label_mask.append(False)

        assert len(input_ids) == _max_seq_length
        assert len(input_mask) == _max_seq_length
        assert len(segment_ids) == _max_seq_length
        assert len(label_ids) == _max_seq_length
        assert len(valid_ids) == _max_seq_length
        assert len(label_mask) == _max_seq_length

        if ex_index < 5:
            logger.info("")
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid_ids,
                          label_mask=label_mask))
    return features


def readfile(filename):
    """
    read file
    """
    f = open(filename)
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []

    return data


class InputExample(object):
    """A single training/test example for simple ner sequence."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    def __init__(self):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    @staticmethod
    def to_json_file(file, data, mode):
        with open(file, mode) as outfile:
            json.dump(data, outfile)
            outfile.write("\n")


if __name__ == '__main__':
    processor = NerProcessor()
    label_list = processor.get_labels()

    num_labels = len(label_list) + 1

    tokenizer = FullTokenizer(os.path.join("model", "vocab.txt"))

    data_path = "datax"
    train_batch_size = 32
    num_train_epochs = 3
    max_seq_length = 20
    seed = 42

    optimizer = None
    ner = None
    train_examples = processor.get_train_examples(data_path)
    num_train_optimization_steps = int(len(train_examples) / train_batch_size) * num_train_epochs

    train_list = list()
    for train in train_examples:
        train_list.append({"words": train.text_a.split(" "), "ner": train.label})

    train_features = convert_examples_to_features(
        train_examples, label_list, max_seq_length, tokenizer)

    logger.info("")
    logger.info("*** Running training ***")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    pprint(train_list)
    for train in train_list:
        NerProcessor.to_json_file("data/train.json", train, "a")
