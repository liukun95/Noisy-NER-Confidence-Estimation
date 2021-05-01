import numpy as np
import torch
from typing import List, Tuple, Set
from common import Instance, Span
import pickle
import torch.optim as optim

import torch.nn as nn

import random
import collections
from config import PAD, ContextEmb, Config
from termcolor import colored
def log_sum_exp_pytorch(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batchSize * from_label * to_label].
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0] ,1 , vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))

def batching_list_instances(config: Config, insts: List[Instance]):
    train_num = len(insts)
    batch_size = config.batch_size
    total_batch = train_num // batch_size + 1 if train_num % batch_size != 0 else train_num // batch_size
    batched_data = []
    
    for batch_id in range(total_batch):
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batched_data.append(simple_batching(config, one_batch_insts))

    return batched_data

def batching_list_instances_marginal(config: Config, insts: List[Instance]):
    train_num = len(insts)
    batch_size = config.batch_size
    total_batch = train_num // batch_size + 1 if train_num % batch_size != 0 else train_num // batch_size
    batched_data = []
    for batch_id in range(total_batch):
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batched_data.append(simple_batching_marginal(config, one_batch_insts))

    return batched_data

def simple_batching(config, insts: List[Instance]) -> Tuple:

    """
    batching these instances together and return tensors. The seq_tensors for word and char contain their word id and char id.
    :return
        word_seq_tensor: Shape: (batch_size, max_seq_length)
        word_seq_len: Shape: (batch_size), the length of each sentence in a batch.
        context_emb_tensor: Shape: (batch_size, max_seq_length, context_emb_size)
        char_seq_tensor: Shape: (batch_size, max_seq_len, max_char_seq_len)
        char_seq_len: Shape: (batch_size, max_seq_len),
        label_seq_tensor: Shape: (batch_size, max_seq_length)
    """
    batch_size = len(insts)
    batch_data = insts
    label_size = config.label_size
    
    word_seq_len = torch.LongTensor(list(map(lambda inst: len(inst.input.words), batch_data)))
    max_seq_len = word_seq_len.max()

    # NOTE: Use 1 here because the CharBiLSTM accepts
    char_seq_len = torch.LongTensor([list(map(len, inst.input.words)) + [1] * (int(max_seq_len) - len(inst.input.words)) for inst in batch_data])
    max_char_seq_len = char_seq_len.max()

    context_emb_tensor = None
    if config.context_emb != ContextEmb.none:
        emb_size = insts[0].elmo_vec.shape[1]
        context_emb_tensor = torch.zeros((batch_size, max_seq_len, emb_size))

    word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    label_seq_tensor =  torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_char_seq_len), dtype=torch.long)

    annotation_mask = None
    if batch_data[0].is_prediction is not None:
        annotation_mask = torch.zeros((batch_size, max_seq_len, label_size), dtype = torch.long)

    gold_label_seq_tensor=None
    if(batch_data[0].gold_output_ids):
        gold_label_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

    for idx in range(batch_size):
        word_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].word_ids)
        if batch_data[idx].output_ids:
            label_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].output_ids)
        if batch_data[idx].gold_output_ids:
            gold_label_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].gold_output_ids)
        
        if config.context_emb != ContextEmb.none:
            context_emb_tensor[idx, :word_seq_len[idx], :] = torch.from_numpy(batch_data[idx].elmo_vec)

        if batch_data[idx].is_prediction is not None:
            for pos in range(len(batch_data[idx].input)):
                if batch_data[idx].is_prediction[pos]:
                    annotation_mask[idx, pos, :] = 1
                    annotation_mask[idx, pos, config.start_label_id] = 0
                    annotation_mask[idx, pos, config.stop_label_id] = 0
                else:
                    annotation_mask[idx, pos, batch_data[idx].output_ids[pos]] = 1
            annotation_mask[idx, word_seq_len[idx]:, :] = 1

        for word_idx in range(word_seq_len[idx]):
            char_seq_tensor[idx, word_idx, :char_seq_len[idx, word_idx]] = torch.LongTensor(batch_data[idx].char_ids[word_idx])
        for wordIdx in range(word_seq_len[idx], max_seq_len):
            char_seq_tensor[idx, wordIdx, 0: 1] = torch.LongTensor([config.char2idx[PAD]]) 

    word_seq_tensor = word_seq_tensor.to(config.device)
    label_seq_tensor = label_seq_tensor.to(config.device)
    gold_label_seq_tensor=gold_label_seq_tensor.to(config.device) if gold_label_seq_tensor is not None else None
    char_seq_tensor = char_seq_tensor.to(config.device)
    word_seq_len = word_seq_len.to(config.device)
    char_seq_len = char_seq_len.to(config.device)
    annotation_mask = annotation_mask.to(config.device) if annotation_mask is not None else None

    return word_seq_tensor, word_seq_len, context_emb_tensor, char_seq_tensor, char_seq_len, annotation_mask, label_seq_tensor,gold_label_seq_tensor

def simple_batching_marginal(config, insts: List[Instance]) -> Tuple:

    """
    batching these instances together and return tensors. The seq_tensors for word and char contain their word id and char id.
    :return 
        word_seq_tensor: Shape: (batch_size, max_seq_length)
        word_seq_len: Shape: (batch_size), the length of each sentence in a batch.
        context_emb_tensor: Shape: (batch_size, max_seq_length, context_emb_size)
        char_seq_tensor: Shape: (batch_size, max_seq_len, max_char_seq_len)
        char_seq_len: Shape: (batch_size, max_seq_len), 
        label_seq_tensor: Shape: (batch_size, max_seq_length)
    """
    batch_size = len(insts)
    batch_data = insts
    label_size = config.label_size
    word_seq_len = torch.LongTensor(list(map(lambda inst: len(inst.input.words), batch_data)))
    max_seq_len = word_seq_len.max()

    # NOTE: Use 1 here because the CharBiLSTM accepts
    char_seq_len = torch.LongTensor([list(map(len, inst.input.words)) + [1] * (int(max_seq_len) - len(inst.input.words)) for inst in batch_data])
    max_char_seq_len = char_seq_len.max()

    context_emb_tensor = None
    if config.context_emb != ContextEmb.none:
        emb_size = insts[0].elmo_vec.shape[1]
        context_emb_tensor = torch.zeros((batch_size, max_seq_len, emb_size))

    word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    label_seq_tensor =  torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_char_seq_len), dtype=torch.long)

    annotation_mask = None
    if batch_data[0].is_prediction is not None:
        annotation_mask = torch.zeros((batch_size, max_seq_len, label_size), dtype = torch.long)

    gold_label_seq_tensor=None
    if(batch_data[0].gold_output_ids):
        gold_label_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

    for idx in range(batch_size):
        word_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].word_ids)
        if batch_data[idx].output_ids:
            label_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].output_ids)
        if batch_data[idx].gold_output_ids:
            gold_label_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].gold_output_ids)
        
        if config.context_emb != ContextEmb.none:
            context_emb_tensor[idx, :word_seq_len[idx], :] = torch.from_numpy(batch_data[idx].elmo_vec)

        if batch_data[idx].is_prediction is not None:
            for pos in range(len(batch_data[idx].input)):
                if batch_data[idx].is_prediction[pos]:
                    annotation_mask[idx, pos, :] = 1
                    
                else:
                    annotation_mask[idx, pos, batch_data[idx].output_ids[pos]] = 1
            annotation_mask[idx, word_seq_len[idx]:, :] = 1

        for word_idx in range(word_seq_len[idx]):
            char_seq_tensor[idx, word_idx, :char_seq_len[idx, word_idx]] = torch.LongTensor(batch_data[idx].char_ids[word_idx])
        for wordIdx in range(word_seq_len[idx], max_seq_len):
            char_seq_tensor[idx, wordIdx, 0: 1] = torch.LongTensor([config.char2idx[PAD]])

    word_seq_tensor = word_seq_tensor.to(config.device)
    label_seq_tensor = label_seq_tensor.to(config.device)
    gold_label_seq_tensor=gold_label_seq_tensor.to(config.device) if gold_label_seq_tensor is not None else None
    
    char_seq_tensor = char_seq_tensor.to(config.device)
    word_seq_len = word_seq_len.to(config.device)
    char_seq_len = char_seq_len.to(config.device)
    annotation_mask = annotation_mask.to(config.device) if annotation_mask is not None else None

    return word_seq_tensor, word_seq_len, context_emb_tensor, char_seq_tensor, char_seq_len, annotation_mask, label_seq_tensor,gold_label_seq_tensor


def lr_decay(config, optimizer: optim.Optimizer, epoch: int) -> optim.Optimizer:
    """
    Method to decay the learning rate
    :param config: configuration
    :param optimizer: optimizer
    :param epoch: epoch number
    :return:
    """
    lr = config.learning_rate / (1 + config.lr_decay * (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning rate is set to: ', lr)
    return optimizer


def load_elmo_vec(file: str, insts: List[Instance]):
    """
    Load the elmo vectors and the vector will be saved within each instance with a member `elmo_vec`
    :param file: the vector files for the ELMo vectors
    :param insts: list of instances
    :return:
    """
    f = open(file, 'rb')
    for inst in insts:
        vec=pickle.load(f)[0][0]
        inst.elmo_vec = vec.detach().numpy()
        size = vec.shape[1]
        assert(vec.shape[0] == len(inst.input.words))
    return size

def get_optimizer(config: Config, model: nn.Module):
    params = model.parameters()
    if config.optimizer.lower() == "sgd":
        print(
            colored("Using SGD: lr is: {}, L2 regularization is: {}".format(config.learning_rate, config.l2), 'yellow'))
        return optim.SGD(params, lr=config.learning_rate, weight_decay=float(config.l2))
    elif config.optimizer.lower() == "adam":
        print(colored("Using Adam", 'yellow'))
        return optim.Adam(params)
    else:
        print("Illegal optimizer: {}".format(config.optimizer))
        exit(1)



def write_results(filename: str, insts):
    f = open(filename, 'w', encoding='utf-8')
    for inst in insts:
        for i in range(len(inst.input)):
            words = inst.input.words
            output = inst.output
            prediction = inst.prediction
            assert len(output) == len(prediction)
            f.write("{}\t{}\t{}\t{}\n".format(i, words[i], output[i], prediction[i]))
        f.write("\n")
    f.close()