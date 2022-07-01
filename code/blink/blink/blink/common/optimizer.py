# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import os
import numpy as np

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch import nn

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import AdamW


patterns_optimizer = {
    'additional_layers': ['additional'],
    'top_layer': ['additional', 'bert_model.encoder.layer.11.'],
    'top4_layers': [
        'additional',
        'bert_model.encoder.layer.11.',
        'encoder.layer.10.',
        'encoder.layer.9.',
        'encoder.layer.8',
    ],
    'all_encoder_layers': ['additional', 'bert_model.encoder.layer'],
    'all': ['additional', 'bert_model.encoder.layer', 'bert_model.embeddings'],
}


def get_bert_optimizer(models, type_optimization, learning_rate, fp16=False):
    """ Optimizes the network with AdamWithDecay
    """
    if type_optimization not in patterns_optimizer:
        print(
            'Error. Type optimizer must be one of %s' % (str(patterns_optimizer.keys()))
        )
    parameters_with_decay = []
    parameters_with_decay_names = []
    parameters_without_decay = []
    parameters_without_decay_names = []
    no_decay = ['bias', 'gamma', 'beta']
    patterns = patterns_optimizer[type_optimization]

    for model in models:
        for n, p in model.named_parameters():
            if any(t in n for t in patterns):
                if any(t in n for t in no_decay):
                    parameters_without_decay.append(p)
                    parameters_without_decay_names.append(n)
                else:
                    parameters_with_decay.append(p)
                    parameters_with_decay_names.append(n)

    print('The following parameters will be optimized WITH decay:')
    print(ellipse(parameters_with_decay_names, 5, ' , '))
    print('The following parameters will be optimized WITHOUT decay:')
    print(ellipse(parameters_without_decay_names, 5, ' , '))

    optimizer_grouped_parameters = [
        {'params': parameters_with_decay, 'weight_decay': 0.01},
        {'params': parameters_without_decay, 'weight_decay': 0.0},
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=learning_rate, 
        correct_bias=False
    )

    if fp16:
        optimizer = fp16_optimizer_wrapper(optimizer)

    return optimizer


def get_type_model_optimizer(models, type_optimization, learning_rate, fp16=False, additional_learning_rate=None):
    """ Optimizes the network with AdamWithDecay
    """

    if additional_learning_rate is None:
        additional_learning_rate = learning_rate

    if type_optimization not in patterns_optimizer:
        print(
            'Error. Type optimizer must be one of %s' % (str(patterns_optimizer.keys()))
        )
    bert_parameters_with_decay = []
    bert_parameters_with_decay_names = []
    bert_parameters_without_decay = []
    bert_parameters_without_decay_names = []

    additional_parameters_with_decay = []
    additional_parameters_with_decay_names = []
    additional_parameters_without_decay = []
    additional_parameters_without_decay_names = []

    no_decay = ['bias', 'gamma', 'beta']
    patterns = patterns_optimizer[type_optimization]

    for model in models:
        for n, p in model.named_parameters():
            # param name doesn't have "additional"
            if any(t in n for t in (set(patterns) - set(["additional"]))):
                if any(t in n for t in no_decay):
                    bert_parameters_without_decay.append(p)
                    bert_parameters_without_decay_names.append(n)
                else:
                    bert_parameters_with_decay.append(p)
                    bert_parameters_with_decay_names.append(n)
            # param name has "additional"
            elif "additional" in n:
                if any(t in n for t in no_decay):
                    additional_parameters_without_decay.append(p)
                    additional_parameters_without_decay_names.append(n)
                else:
                    additional_parameters_with_decay.append(p)
                    additional_parameters_with_decay_names.append(n)


    # print('The following parameters will be optimized WITH decay:')
    # print(ellipse(parameters_with_decay_names, 5, ' , '))
    # print('The following parameters will be optimized WITHOUT decay:')
    # print(ellipse(parameters_without_decay_names, 5, ' , '))

    optimizer_grouped_parameters = [
        {'params': bert_parameters_with_decay, 'weight_decay': 0.01},
        {'params': bert_parameters_without_decay, 'weight_decay': 0.0},
        {'params': additional_parameters_with_decay, 'weight_decay': 0.01, 'lr': additional_learning_rate},
        {'params': additional_parameters_without_decay, 'weight_decay': 0.0, 'lr': additional_learning_rate},
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        correct_bias=False
    )

    if fp16:
        optimizer = fp16_optimizer_wrapper(optimizer)

    return optimizer


def ellipse(lst, max_display=5, sep='|'):
    """
    Like join, but possibly inserts an ellipsis.
    :param lst: The list to join on
    :param int max_display: the number of items to display for ellipsing.
        If -1, shows all items
    :param string sep: the delimiter to join on
    """
    # copy the list (or force it to a list if it's a set)
    choices = list(lst)
    # insert the ellipsis if necessary
    if max_display > 0 and len(choices) > max_display:
        ellipsis = '...and {} more'.format(len(choices) - max_display)
        choices = choices[:max_display] + [ellipsis]
    return sep.join(str(c) for c in choices)
