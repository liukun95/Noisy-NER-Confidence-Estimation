import torch
import torch.nn as nn
from config import START, STOP, PAD, log_sum_exp_pytorch
from model.charbilstm import CharBiLSTM
from model.bilstm_encoder import BiLSTMEncoder
from model.linear_partial_crf_inferencer import LinearCRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import ContextEmb
from typing import Tuple
from overrides import overrides
import numpy as np

def check_remove_ratio(gold_tags,tags,small_loss_mask,mask,negative_mask):
    remove_mask=(1-small_loss_mask)*mask
    positive_mask=(1-negative_mask)*mask
    clean_mask=torch.eq(gold_tags,tags).float()*mask
    noise_mask=(1-clean_mask)*mask.float()
    remove_neg=remove_mask*negative_mask
    remove_right_neg=noise_mask*remove_neg
    remove_pos=remove_mask*(1-negative_mask)*mask
    remove_right_pos=noise_mask*remove_pos
    noise_positive=noise_mask*positive_mask*mask
    noise_negative=noise_mask*negative_mask*mask
    neg_recall=remove_right_neg.sum() / (noise_negative.sum()+1e-8)
    pos_recall=remove_right_pos.sum() / (noise_positive.sum()+1e-8)
    neg_precision=remove_right_neg.sum() / (remove_neg.sum()+1e-8)
    pos_precision=remove_right_pos.sum() / (remove_pos.sum()+1e-8)
    neg_f1=2*neg_recall*neg_precision/(neg_recall+neg_precision+1e-8)
    pos_f1=2*pos_recall*pos_precision/(pos_recall+pos_precision+1e-8)
    
    
    return neg_recall.item(),pos_recall.item(),neg_precision.item(),pos_precision.item(),neg_f1.item(),pos_f1.item()

def gen_dic(labels,label2idx):
    types=set()
    for label in labels:
        if(label.startswith('B') or label.startswith('S') or label.startswith('E') or label.startswith('I')):
            tp=label.split('-')[1]
            types.add(tp)
    pos_dic={'O':[label2idx['O']]}
    type_dic={'O':[label2idx['O']]}
    for label in labels:
        if(label=='O' or label.startswith('<')):
            continue
        pos,type=label.split('-')[0],label.split('-')[1]
        if(pos in pos_dic):
            pos_dic[pos].append(label2idx[label])
        else:
            pos_dic[pos]=[label2idx[label]]
        if(type in type_dic):
            type_dic[type].append(label2idx[label])
        else:
            type_dic[type]=[label2idx[label]]
    for tp in types:
        type_dic[tp].append(label2idx['O'])
    for pos in ['B','I','E','S']:
        pos_dic[pos].append(label2idx['O'])
    return pos_dic,type_dic

def gen_embedding_table(idx2label,type_dic,pos_dic):
    type_embedding=torch.zeros(len(idx2label),len(idx2label))
    pos_embedding=torch.zeros(len(idx2label),len(idx2label))
    #type_embedding
    for id,label in enumerate(idx2label):
        
        if(label.startswith('B') or label.startswith('S') or label.startswith('E') or label.startswith('I')):
            indexes=type_dic[label.split('-')[1]]
            for index in indexes:
                type_embedding[id][index]=1
        elif(label=='O'):
            type_embedding[id]=torch.ones_like(type_embedding[id])
            
    #pos_embedding
    for id,label in enumerate(idx2label):
        
        if(label.startswith('B') or label.startswith('S') or label.startswith('E') or label.startswith('I')):
            indexes=pos_dic[label.split('-')[0]]
            for index in indexes:
                pos_embedding[id][index]=1
        elif(label=='O'):
            pos_embedding[id]=torch.ones_like(pos_embedding[id])
            
    type_embedding,pos_embedding =pos_embedding,type_embedding
    return type_embedding,pos_embedding



class NNCRF_sl(nn.Module):

    def __init__(self, config, print_info: bool = True):
        super(NNCRF_sl, self).__init__()
        self.device = config.device
        self.encoder = BiLSTMEncoder(config, print_info=print_info)
        self.inferencer = LinearCRF(config, print_info=print_info)
        self.label2idx = config.label2idx
        self.idx2word=config.idx2word
        self.idx2labels=config.idx2labels
        self.Oid = self.label2idx['O']
        self.padid = self.label2idx['<PAD>']
        self.startid=self.label2idx['<START>']
        self.stopid=self.label2idx['<STOP>']
        
        
        self.pos_dic, self.type_dic=gen_dic(config.label2idx.keys(),self.label2idx)
        
        self.tags_num=len(self.idx2labels)
        e_type,pos=gen_embedding_table(self.idx2labels,self.type_dic,self.pos_dic)
        self.type_embedding = torch.nn.Embedding(self.tags_num, self.tags_num).from_pretrained(e_type,freeze=True).cuda(self.device)
        self.pos_embedding=torch.nn.Embedding(self.tags_num, self.tags_num).from_pretrained(pos,freeze=True).cuda(self.device)
        
    @overrides
    def forward(self, words: torch.Tensor,
                    word_seq_lens: torch.Tensor,
                    batch_context_emb: torch.Tensor,
                    chars: torch.Tensor,
                    char_seq_lens: torch.Tensor,
                    annotation_mask : torch.Tensor,
                    tags: torch.Tensor,
                    gold_tags=None,
                    forget_rate_neg=0,forget_rate_pos=0,is_constrain=False) -> torch.Tensor:
        """
        Calculate the negative loglikelihood.
        :param words: (batch_size x max_seq_len)
        :param word_seq_lens: (batch_size)
        :param batch_context_emb: (batch_size x max_seq_len x context_emb_size)
        :param chars: (batch_size x max_seq_len x max_char_len)
        :param char_seq_lens: (batch_size x max_seq_len)
        :param tags: (batch_size x max_seq_len)
        :return: the loss with shape (batch_size)
        """
        
        lstm_scores= self.encoder(words, word_seq_lens, batch_context_emb, chars, char_seq_lens)
        batch_size = words.size(0)
        sent_len = words.size(1)
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long).view(1, sent_len).expand(batch_size, sent_len).to(self.device)
        mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len)).to(self.device).float()

        onehot_label = torch.zeros_like(lstm_scores).scatter_(-1, tags.unsqueeze(-1), 1)
        log_marginals = self.inferencer.marginal(lstm_scores,word_seq_lens)
        token_prob = log_marginals
        forward_loss = -(onehot_label * token_prob).sum(dim=-1) * mask
        forward_loss=forward_loss.detach()
        negative_mask=torch.eq(tags, self.Oid).float()* mask.float()
        positive_mask =(1 - negative_mask)*mask.float()


        tmp = forward_loss.view(batch_size * sent_len) + (1000 * (1 - mask).view(batch_size * sent_len)) + (
                    1000 * (positive_mask.view(batch_size * sent_len)))
        
        index=torch.argsort(tmp, dim=-1)
        remember_rate_neg = 1.0 - forget_rate_neg
        
        num_remember = int(remember_rate_neg * (negative_mask.sum()))
        small_loss_index = index[:num_remember]
        small_loss_mask_neg = torch.zeros_like(tmp)
        for num in small_loss_index:
            small_loss_mask_neg[num] = 1
        small_loss_mask_neg = small_loss_mask_neg.view((batch_size, sent_len))
        if num_remember == 0:
            small_loss_mask_neg = negative_mask
        remove_num_neg = negative_mask.sum() - small_loss_mask_neg.sum()
        
        tmp = forward_loss.view(batch_size * sent_len) + (1000 * (1 - mask).view(batch_size * sent_len)) + (
                    1000 * (negative_mask.view(batch_size * sent_len)))
        
        index=torch.argsort(tmp, dim=-1)
        remember_rate_pos = 1.0 - forget_rate_pos
        
        num_remember = int(remember_rate_pos * (positive_mask.sum()))
        small_loss_index = index[:num_remember]
        small_loss_mask_pos = torch.zeros_like(tmp)
        for num in small_loss_index:
            small_loss_mask_pos[num] = 1
        small_loss_mask_pos = small_loss_mask_pos.view((batch_size, sent_len))
        if num_remember == 0:
            small_loss_mask_pos = positive_mask

        small_loss_mask = (small_loss_mask_pos.bool() + small_loss_mask_neg.bool()).float()
        small_loss_mask = small_loss_mask.detach()
        
        if(gold_tags!=None): 
            neg_recall,pos_recall,neg_precision,pos_precision,neg_f1,pos_f1=check_remove_ratio(gold_tags,tags,small_loss_mask,mask,negative_mask)
         
        
        partial_label=torch.ones_like(onehot_label)
        
        
        type_lookup = self.type_embedding(tags)
        pos_lookup=self.pos_embedding(tags)
         
        prob = log_marginals.exp()
        prob=prob.detach()
        type_prob=(prob*type_lookup).mean(dim=-1)
        pos_prob=(prob*pos_lookup).mean(dim=-1)
        type_change_mask=(type_prob>pos_prob)*mask*(1-small_loss_mask)
        pos_change_mask=(type_prob<pos_prob)*mask*(1-small_loss_mask)
        
        change_label=((type_change_mask.unsqueeze(-1)*type_lookup)+(pos_change_mask.unsqueeze(-1)*pos_lookup))+((1-small_loss_mask)*(1-type_change_mask)*(1-pos_change_mask)).unsqueeze(-1)*partial_label

        if(is_constrain):
            label_tag_mask=(small_loss_mask.unsqueeze(-1)*onehot_label) + ((1-small_loss_mask).unsqueeze(-1)*change_label)
        else:
            label_tag_mask = small_loss_mask.unsqueeze(-1) * onehot_label + (1 - small_loss_mask).unsqueeze(-1) * partial_label
        
        label_tag_mask=label_tag_mask.detach()
        unlabed_score, labeled_score = self.inferencer(lstm_scores, word_seq_lens,label_tag_mask)

        loss_neg=(forward_loss*negative_mask).sum()/(negative_mask.sum()+1e-6)
        loss_pos=(forward_loss*positive_mask).sum()/(positive_mask.sum()+1e-6)
        
        if(gold_tags!=None):
            return unlabed_score - labeled_score,[neg_recall,pos_recall,neg_precision,pos_precision,neg_f1,pos_f1],loss_neg,loss_pos
        else:
            return unlabed_score - labeled_score

    def decode(self, batchInput: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        wordSeqTensor, wordSeqLengths, batch_context_emb, charSeqTensor, charSeqLengths, annotation_mask, tagSeqTensor,_= batchInput
        
        features = self.encoder(wordSeqTensor, wordSeqLengths, batch_context_emb,charSeqTensor,charSeqLengths)
        bestScores, decodeIdx = self.inferencer.decode(features, wordSeqLengths, annotation_mask)
        
        return bestScores, decodeIdx

