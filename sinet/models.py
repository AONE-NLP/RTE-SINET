import torch
from torch import nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel, BertTokenizer

from sinet import sampling
from sinet import util
import math
import random
from collections import OrderedDict
import json
import numpy as np
from torch.autograd import Variable
from math import sqrt


from typing import Dict, Optional
import torch.nn.functional as F
from torch import Tensor


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]                  #768

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h

def del_tensor_ele_n(arr, index, n):
    """
    arr: 输入tensor
    index: 需要删除位置的索引
    n: 从index开始，需要删除的行数
    """
    arr1 = arr[:,0:index,:]
    arr2 = arr[:,index+n:,:]
    return torch.cat((arr1,arr2),dim=1)

class CrossMultiAttention(nn.Module):
    def __init__(self, dim, num_heads=6, attn_drop=0.2, proj_drop=0.2, qkv_bias=False, qk_scale=None):
        super().__init__()

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.wq = nn.Linear(dim, dim, bias=qkv_bias).to(_device)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias).to(_device)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias).to(_device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.addnorm = AddNorm([1, dim])

    def forward(self, y, x_cls):

        y_all = torch.concat((y, x_cls), dim=1)
        B, N, C = y_all.shape
        
        q = self.wq(x_cls).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(y_all).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(y_all).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # print(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        o = x+x_cls

        return self.addnorm(o)

class AddNorm(nn.Module):
    """layer normalization"""
    def __init__(self, normalized_shape, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, Y):
        return self.ln(Y)

class SiNET(BertPreTrainedModel):
    """ Span-based model to jointly extract entities and relations """

    VERSION = '1.1'

    def __init__(self, config: BertConfig, cls_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, head: int,  freeze_transformer: bool, max_pairs: int = 100):
        super(SiNET, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)

        # TAP layers
        self.linear_r = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.linear_s = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.cross_attention = CrossMultiAttention(config.hidden_size, head)

        self.rel_classifier = nn.Linear(config.hidden_size * 4 + size_embedding*2, relation_types)
        self.entity_classifier = nn.Linear(config.hidden_size * 2 + size_embedding, entity_types)
        self.size_embeddings = nn.Embedding(100, size_embedding)
        self.dropout = nn.Dropout(prop_drop)

        self._cls_token = cls_token
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs

        # weight initialization
        self.init_weights()

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False


    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, relations: torch.tensor, rel_masks: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        batch_size = encodings.shape[0]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, e_r, cross_cls = self._classify_entities(encodings, h, entity_masks, size_embeddings)

        # classify relations
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(e_r, cross_cls, size_embeddings,
                                                        relations, rel_masks, h_large, h, i)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits

        return entity_clf, rel_clf

    def _forward_inference(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                           entity_sizes: torch.tensor, entity_spans: torch.tensor, entity_sample_masks: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']

        batch_size = encodings.shape[0]
        ctx_size = context_masks.shape[-1]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes

        entity_clf, e_r, cross_cls = self._classify_entities(encodings, h, entity_masks, size_embeddings)

        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        relations, rel_masks, rel_sample_masks = self._filter_spans(entity_clf, entity_spans,
                                                                    entity_sample_masks, ctx_size)

        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(e_r, cross_cls, size_embeddings,
                                                        relations, rel_masks, h_large, h, i)
            # apply sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        rel_clf = rel_clf * rel_sample_masks  # mask

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, rel_clf, relations

    def _classify_entities(self, encodings, h, entity_masks, size_embeddings):
        h_s_all = self.linear_s(h)
        h_r_all = self.linear_r(h)
       

        h_s_cls = get_token(h_s_all, encodings, self._cls_token).unsqueeze(dim=1)
        h_r_cls = get_token(h_r_all, encodings, self._cls_token).unsqueeze(dim=1)
       
        h_s = del_tensor_ele_n(h_s_all, 0, 1)       
        h_r = del_tensor_ele_n(h_r_all, 0, 1)
        

        #do cross-task-attention
        cross_cls_s = self.cross_attention(h_r, h_s_cls)
        cross_cls_r = self.cross_attention(h_s, h_r_cls)


        #get entity candidate spans
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)

        entity_spans = m + h_s_all.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        relation_spans = m + h_r_all.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)


        # max pool entity candidate spans
        entity_spans_pool = entity_spans.max(dim=2)[0]
        relation_spans_pool = relation_spans.max(dim=2)[0]

        ner_ctx = cross_cls_s.repeat(1, entity_spans_pool.shape[1], 1)
        
        entity_repr = torch.cat([ner_ctx, entity_spans_pool, size_embeddings], dim=2)  
        entity_repr = self.dropout(entity_repr)

        # classify entity candidates
        entity_clf = self.entity_classifier(entity_repr)

        return entity_clf, relation_spans_pool, cross_cls_r

    def _classify_relations(self, e_r, cross_cls, size_embeddings, relations, rel_masks, h_large, h, chunk_start):
        batch_size = relations.shape[0]

        # create chunks if necessary
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h_large = h_large[:, :relations.shape[1], :]

       
        # get pairs of entity candidate representations
        entity_pairs = util.batch_index(e_r, relations)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)


        # get corresponding size embeddings
        size_pair_embeddings = util.batch_index(size_embeddings, relations)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # relation context (context between entity candidate pair)
        # mask non entity candidate tokens
        m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)
        rel_ctx = m + h_large
        # max pooling
        rel_ctx = rel_ctx.max(dim=2)[0]
        # set the context vector of neighboring or adjacent entity candidates to zero
        rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0


        rel_cls = cross_cls.repeat(1, size_pair_embeddings.shape[1], 1)

        # create relation candidate representations including context, max pooled entity candidate pairs
        # and corresponding size embeddings
        rel_repr = torch.cat([rel_ctx, rel_cls, entity_pairs, size_pair_embeddings], dim=2)
        rel_repr = self.dropout(rel_repr)

        # classify relation candidates
        chunk_rel_logits = self.rel_classifier(rel_repr)
        return chunk_rel_logits

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()  # get entity type (including none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # get spans classified as entities
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()

            # create relations and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.rel_classifier.weight.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks

    def forward(self, *args, inference=False, **kwargs):
        if not inference:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_inference(*args, **kwargs)


# Model access

_MODELS = {
    'sinet': SiNET,
}


def get_model(name):
    return _MODELS[name]
