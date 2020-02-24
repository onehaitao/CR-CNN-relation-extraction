#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class CRCNN(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super().__init__()
        self.word_vec = word_vec
        self.class_num = class_num
        self.device = config.device

        # hyper parameters and others
        self.max_len = config.max_len
        self.word_dim = config.word_dim
        self.pos_dim = config.pos_dim
        self.pos_dis = config.pos_dis

        self.dropout_value = config.dropout
        self.filter_num = config.filter_num
        self.window = config.window

        self.dim = self.word_dim + 2 * self.pos_dim

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.filter_num,
            kernel_size=(self.window, self.dim),
            stride=(1, 1),
            bias=True,
            padding=(1, 0),  # same padding
            padding_mode='zeros'
        )
        self.maxpool = nn.MaxPool2d((self.max_len, 1))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_value)
        # self.dense = nn.Linear(
        #     in_features=self.filter_num,
        #     out_features=self.class_num,
        #     bias=True
        # )
        self.r = (6/(self.class_num + self.filter_num))**(0.5)
        self.relation_weight = nn.Parameter(2 * self.r * (torch.rand(self.filter_num, self.class_num) - 0.5))

        # initialize weight
        init.xavier_normal_(self.pos1_embedding.weight)
        init.xavier_normal_(self.pos2_embedding.weight)
        init.xavier_normal_(self.conv.weight)
        init.constant_(self.conv.bias, 0.)
        # init.xavier_normal_(self.dense.weight)
        # init.constant_(self.dense.bias, 0.)

    def encoder_layer(self, token, pos1, pos2):
        word_emb = self.word_embedding(token)  # B*L*word_dim
        pos1_emb = self.pos1_embedding(pos1)  # B*L*pos_dim
        pos2_emb = self.pos2_embedding(pos2)  # B*L*pos_dim
        emb = torch.cat(tensors=[word_emb, pos1_emb, pos2_emb], dim=-1)
        return emb  # B*L*D, D=word_dim+2*pos_dim

    def conv_layer(self, emb, mask):
        emb = emb.unsqueeze(dim=1)  # B*1*L*D
        conv = self.conv(emb)  # B*C*L*1

        # mask, remove the effect of 'PAD'
        conv = conv.view(-1, self.filter_num, self.max_len)  # B*C*L
        mask = mask.unsqueeze(dim=1)  # B*1*L
        mask = mask.expand(-1, self.filter_num, -1)  # B*C*L
        conv = conv.masked_fill_(mask.eq(0), float('-inf'))  # B*C*L
        conv = conv.unsqueeze(dim=-1)  # B*C*L*1
        return conv

    def single_maxpool_layer(self, conv):
        pool = self.maxpool(conv)  # B*C*1*1
        pool = pool.view(-1, self.filter_num)  # B*c
        return pool

    def forward(self, data):
        token = data[:, 0, :].view(-1, self.max_len)
        pos1 = data[:, 1, :].view(-1, self.max_len)
        pos2 = data[:, 2, :].view(-1, self.max_len)
        mask = data[:, 3, :].view(-1, self.max_len)
        emb = self.encoder_layer(token, pos1, pos2)
        # emb = self.dropout(emb)
        conv = self.conv_layer(emb, mask)
        # conv = self.relu(conv)
        pool = self.single_maxpool_layer(conv)
        # sentence_feature = self.linear(pool)
        # sentence_feature = self.tanh(sentence_feature)
        # sentence_feature = self.dropout(sentence_feature)
        # logits = self.dense(sentence_feature)
        feature = self.dropout(pool)
        feature = self.tanh(feature)
        scores = torch.mm(feature, self.relation_weight)
        return scores


class RankingLoss(nn.Module):
    def __init__(self, class_num, config):
        super(RankingLoss, self).__init__()
        self.class_num = class_num
        self.margin_positive = config.margin_positive
        self.margin_negative = config.margin_negative
        self.gamma = config.gamma
        self.device = config.device

    def forward(self, scores, labels):
        labels = labels.view(-1, 1)
        positive_mask = (torch.ones([labels.shape[0], self.class_num], device=self.device)
                         * float('inf')).scatter_(1, labels, 0.0)
        negative_mask = torch.zeros([labels.shape[0], self.class_num], device=self.device).scatter_(1, labels, float('inf'))
        positive_scores, _ = torch.max(scores-positive_mask, dim=1)
        negative_scores, _ = torch.max(scores-negative_mask, dim=1)
        positive_loss = torch.log1p(torch.exp(self.gamma*(self.margin_positive-positive_scores)))
        positive_loss[labels[:, 0] == 0] = 0.0
        negative_loss = torch.log1p(torch.exp(self.gamma*(self.margin_negative+negative_scores)))
        loss = torch.mean(positive_loss + negative_loss)
        return loss
