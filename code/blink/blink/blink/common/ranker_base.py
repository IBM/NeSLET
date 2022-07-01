# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torch import nn
import torch


def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model


class BertEncoder(nn.Module):
    def __init__(self, bert_model, output_dim, layer_pulled=-1, add_linear=None):
        super(BertEncoder, self).__init__()
        self.layer_pulled = layer_pulled
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        self.bert_model = bert_model
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None

    def forward(self, token_ids, segment_ids, attention_mask):
        output_bert, output_pooler = self.bert_model(
            token_ids, segment_ids, attention_mask
        )
        # get embedding of [CLS] token
        if self.additional_linear is not None:
            embeddings = output_pooler
        else:
            embeddings = output_bert[:, 0, :]

        # in case of dimensionality reduction
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings))
        else:
            result = embeddings

        return result


class BertEncoder2(nn.Module):
    def __init__(self, bert_model, output_dim, layer_pulled=-1, add_linear=None, cls_start=1, cls_end=12):
        super(BertEncoder2, self).__init__()
        self.layer_pulled = layer_pulled
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        self.bert_model = bert_model
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None

        self.cls_start = cls_start
        self.cls_end = cls_end

        assert 0 <= cls_start <= cls_end

        if cls_start < cls_end:
            cls_weights = torch.randn(cls_end - cls_start)
            self.additional_cls_weights = nn.Parameter(cls_weights)

    def forward(self, token_ids, segment_ids, attention_mask):
        output_bert, output_pooler, all_layers = self.bert_model(
            token_ids, segment_ids, attention_mask
        )
        # get embedding of [CLS] token
        if self.additional_linear is not None:
            embeddings = output_pooler
        else:
            embeddings = output_bert[:, 0, :]

        # in case of dimensionality reduction
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings))
        else:
            result = embeddings

        if self.cls_start < self.cls_end:
            all_cls_vectors = []
            for layer_l in all_layers[self.cls_start: self.cls_end]:
                cls_vec_l = layer_l[:, 0, :]
                all_cls_vectors.append(cls_vec_l.unsqueeze(1))

            all_cls_vectors = torch.cat(all_cls_vectors, dim=1)

            assert len(all_cls_vectors.shape) == 3
            assert all_cls_vectors.shape[0] == token_ids.shape[0]
            assert all_cls_vectors.shape[1] == self.cls_end - self.cls_start
            assert all_cls_vectors.shape[2] == all_layers[0].shape[2]

            vector_for_type_prediction = torch.matmul(all_cls_vectors.permute(0, 2, 1),
                                                      self.additional_cls_weights.unsqueeze(0).permute(1, 0)).permute(0,
                                                                                                                      2,
                                                                                                                      1).squeeze(
                1)
        else:
            vector_for_type_prediction = all_layers[self.cls_start][:, 0, :]

        assert len(vector_for_type_prediction.shape) == 2
        assert vector_for_type_prediction.shape[0] == token_ids.shape[0]
        assert vector_for_type_prediction.shape[1] == all_layers[0].shape[2]

        return {"for_entity_prediction": result, "for_type_prediction": vector_for_type_prediction}
