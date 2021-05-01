import torch
import pdb


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, label=None,gold_label=None):
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
        self.label = label
        self.gold_label=gold_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids,gold_label_ids, gather_ids,gather_masks,annotation_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.gold_label_ids=gold_label_ids
        self.gather_ids=gather_ids
        self.gather_masks=gather_masks
        self.annotation_mask=annotation_mask



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, logger):
        self.logger = logger

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
    def _read(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines


class Processor(DataProcessor):
    """Processor NQG data set."""

    def __init__(self, logger):
        self.logger = logger
        self.labels = list()
        self.labels.append("O")

    def get_train_examples(self, input_file):
        """See base class."""
        self.logger.info("LOOKING AT {}".format(input_file))
        return self._create_examples(
            self._read(input_file), "train")

    def get_dev_examples(self, input_file):
        """See base class."""
        self.logger.info("LOOKING AT {}".format(input_file))
        return self._create_examples(
            self._read(input_file), "dev")

    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples_ori(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        tokens = []
        labels = []
        for (i, line) in enumerate(lines):
            if not line:
                examples.append(
                    InputExample(guid=len(examples), text_a=tokens, text_b=None, label=labels))
                tokens = []
                labels = []
            else:
                sp = line.split(' ')
                tokens.append(sp[0])
                label = sp[-1]
                labels.append(label)
                if label not in self.labels:
                    self.labels.append(label)

        if len(tokens) > 0:
            examples.append(
                InputExample(guid=len(examples), text_a=tokens, text_b=None, label=labels))
        return examples


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        tokens = []
        labels = []
        gold_labels=[]
        for (i, line) in enumerate(lines):
            if not line:
                examples.append(
                    InputExample(guid=len(examples), text_a=tokens, label=labels,gold_label=gold_labels))
                tokens = []
                labels = []
                gold_labels = []
            else:
                sp = line.split(' ')
                tokens.append(sp[0])
                label = sp[1]
                if(len(sp)>2):
                    gold_labels.append(sp[2])
                labels.append(label)
                if label not in self.labels:
                    self.labels.append(label)

        if len(tokens) > 0:
            examples.append(
                InputExample(guid=len(examples), text_a=tokens, label=labels,gold_label=gold_labels))
        return examples

    def BIO2BIOES(self,output):
        for pos in range(len(output)):
            
            curr_entity = output[pos]
            if pos == len(output) - 1:
                if curr_entity.startswith('B'):
                    output[pos]='S'+curr_entity[1:] 
                elif curr_entity.startswith('I'):
                    output[pos]='E'+curr_entity[1:]
            else:
                next_entity = output[pos + 1]
                if curr_entity.startswith('B'):
                    if next_entity.startswith('O') or next_entity.startswith('B'):
                        output[pos] = 'S'+curr_entity[1:]
                elif curr_entity.startswith('I'):
                    if next_entity.startswith('O') or next_entity.startswith('B'):
                        output[pos]='E'+curr_entity[1:]
        
        return output
    def convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer,is_train):
        """Loads a data file into a list of `InputBatch`s."""
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []

        for (ex_index, example) in enumerate(examples):
            
            tokens = tokenizer.tokenize(' '.join(example.text_a))
            
            if(len(tokens)>max_seq_length):
                print("remove "+' '.join(example.text_a))
                continue

            labels = self.BIO2BIOES(example.label)
            gold_labels=self.BIO2BIOES(example.gold_label)
            gather_ids = list()
            
            for (idx, token) in enumerate(tokens):
                if (not token.startswith("##") and idx < max_seq_length - 2):
                    gather_ids.append(idx + 1)
            annotation_mask=torch.ones((max_seq_length, len(label_list)), dtype=torch.long)
            if(is_train):
                annotation_mask=torch.zeros((max_seq_length, len(label_list)), dtype=torch.long)
                for pos,label in enumerate(labels):
                    if(label=='O'):
                        
                        annotation_mask[pos,:]=1
                        annotation_mask[pos, label_map["<START>"]] = 0
                        annotation_mask[pos, label_map["<STOP>"]] = 0
                    else:
                        annotation_mask[pos,label_map[label]]=1
                annotation_mask[len(labels):,:]=1

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:max_seq_length - 2]
            
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            label_ids=[label_map[label] for label in labels]
            gold_label_ids=[label_map[label] for label in gold_labels]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            
            
            label_padding=[label_map["<PAD>"]]*(max_seq_length-len(label_ids))
            label_ids+=label_padding
            gold_label_ids+=label_padding

            gather_padding = [0] * (max_seq_length - len(gather_ids))
            gather_masks = [1] * len(gather_ids) + gather_padding
            gather_ids += gather_padding
            
            
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(gather_ids) == max_seq_length
            assert len(gather_masks) == max_seq_length

            if ex_index < 2:
                self.logger.info("*** Example ***")
                self.logger.info("guid: %s" % (example.guid))
                self.logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                self.logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                self.logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                self.logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                self.logger.info(
                    "label_ids: %s" % " ".join([str(x) for x in label_ids]))
                self.logger.info(
                    "gold_label_ids: %s" % " ".join([str(x) for x in gold_label_ids]))
                self.logger.info(
                    "gather_ids: %s" % " ".join([str(x) for x in gather_ids]))
                self.logger.info(
                    "gather_masks: %s" % " ".join([str(x) for x in gather_masks]))

            features.append(InputFeatures(input_ids=input_ids,input_mask=input_mask,segment_ids=segment_ids,label_ids=label_ids,gold_label_ids=gold_label_ids,gather_ids=gather_ids,gather_masks=gather_masks,annotation_mask=annotation_mask))
        return features

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()


def get_result(results,step=None):
    
    correct, precision_all, recall_all = 0, 0, 0
    output_str = []
    predict_spans = []
    for item in results:

        tokens = item["token"]
        golds = item["gold"]
        preds = item["pred"]
        new_words = list()
        new_golds = list()
        new_preds = list()
        for idx, word in enumerate(tokens):
            if (word == "[CLS]" or word == "[SEP]"):
                continue
            if (not word.startswith("##")):
                new_words.append(word)
        new_golds=golds
        new_preds=preds

        output_str.append("\n".join(
            [word + " " + gold + " " + pred + " "  for word, gold, pred in
             zip(new_words, new_golds, new_preds)]))

        def get_spans_ori(labels):
            spans = set()
            i = 0
            while True:
                if i == len(labels):
                    break
                if labels[i].startswith('B'):
                    label = labels[i][2:]
                    j = i
                    while True:
                        j = j + 1
                        if j == len(labels) or labels[j].startswith('B') or \
                                labels[j].startswith('O'):
                            spans.add(str(i) + "_" + str(j - 1) + "_" + label)
                            break
                        else:
                            if labels[j][2:] == label:
                                continue
                            else:
                                break
                    i = j
                else:
                    i += 1
            return spans
        def get_spans(output):
            output_spans=set()
            start = -1
            for i in range(len(output)):
                if output[i].startswith("B-"):
                    start = i
                if output[i].startswith("E-"):
                    end = i
                    output_spans.add(str(start) + "_" + str(end) + "_" + output[i][2:])
                if output[i].startswith("S-"):
                    output_spans.add(str(i) + "_" + str(i) + "_" + output[i][2:])
            return output_spans

        gold_set = get_spans(new_golds)
        predict_set = get_spans(new_preds)
        predict_spans.append(predict_set)
        correct_set = predict_set.intersection(gold_set)
        correct += len(correct_set)
        precision_all += len(predict_set)
        recall_all += len(gold_set)

    precision = recall = 0
    if precision_all > 0:
        precision = correct / precision_all
    if recall_all > 0:
        recall = correct / recall_all

    if precision == 0 or recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return precision, recall, F, output_str, predict_spans


def categoricalAccuracy(predictions: torch.Tensor,
                        gold_labels: torch.Tensor,
                        mask: torch.Tensor = None,
                        tie_break=False,
                        top_k=1):
    total_count = 0
    correct_count = 0

    num_classes = predictions.size(-1)
    if gold_labels.dim() != predictions.dim() - 1:
        raise ValueError("gold_labels must have dimension == predictions.size() - 1 but "
                         "found tensor of shape: {}".format(predictions.size()))
    if (gold_labels >= num_classes).any():
        raise ValueError("A gold label passed to Categorical Accuracy contains an id >= {}, "
                         "the number of classes.".format(num_classes))

    predictions = predictions.view((-1, num_classes))
    gold_labels = gold_labels.view(-1).long()
    if not tie_break:
        # Top K indexes of the predictions (or fewer, if there aren't K of them).
        # Special case topk == 1, because it's common and .max() is much faster than .topk().
        if top_k == 1:
            top_k = predictions.max(-1)[1].unsqueeze(-1)
        else:
            top_k = predictions.topk(min(top_k, predictions.shape[-1]), -1)[1]

        # This is of shape (batch_size, ..., top_k).
        correct = top_k.eq(gold_labels.unsqueeze(-1)).float()
    else:
        # prediction is correct if gold label falls on any of the max scores. distribute score by tie_counts
        max_predictions = predictions.max(-1)[0]
        max_predictions_mask = predictions.eq(max_predictions.unsqueeze(-1))
        # max_predictions_mask is (rows X num_classes) and gold_labels is (batch_size)
        # ith entry in gold_labels points to index (0-num_classes) for ith row in max_predictions
        # For each row check if index pointed by gold_label is was 1 or not (among max scored classes)
        correct = max_predictions_mask[torch.arange(gold_labels.numel()).long(), gold_labels].float()
        tie_counts = max_predictions_mask.sum(-1)
        correct /= tie_counts.float()
        correct.unsqueeze_(-1)

    if mask is not None:
        correct *= mask.view(-1, 1).float()
        total_count += mask.sum()
    else:
        total_count += gold_labels.numel()
    correct_count += correct.sum()
    return float(correct_count), float(total_count)
