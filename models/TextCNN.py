# encoding: utf-8
"""
@author: vinklibrary
@contact: shenxvdong1@gmail.com
@site: Todo

@version: 1.0
@license: None
@file: TextCNN.py
@time: 2019/7/17 16:20

Yoon Kim  《(2014 EMNLP) Convolutional Neural Networks for Sentence Classification》
"""
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, seq_length, num_classes, vocabulary_size , embedding_size, filter_size_arr ):
        super(TextCNN, self).__init__()

        self.seq_length = seq_length
        self.num_classes = num_classes
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.filter_size_arr = filter_size_arr
        pass

    def forward(self, *input):
        pass

