# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)

from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.common.ranker_base import BertEncoder, BertEncoder2, get_model_obj
from blink.common.optimizer import get_bert_optimizer
from blink.biencoder.ontology_nn import OntologyNN


def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


def safe_divide(numerator, denominator):
    if denominator == 0:
        denominator = 1

    return numerator / denominator


def get_aux_task_weight(training_progress, gamma=10):
    return (2 / (1 + (math.e ** (-gamma * training_progress)))) - 1


class GradientThrottle(torch.autograd.Function):
    scale = 1.0

    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return GradientThrottle.scale * grad_output


def grad_throttle(x, scale=1.0):
    GradientThrottle.scale = scale
    return GradientThrottle.apply(x)


def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderModule, self).__init__()
        ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        cand_bert = BertModel.from_pretrained(params["bert_model"])
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = ctxt_bert.config

    def forward(
            self,
            token_idx_ctxt,
            segment_idx_ctxt,
            mask_ctxt,
            token_idx_cands,
            segment_idx_cands,
            mask_cands,
    ):
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_ctxt, embedding_cands


class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(BiEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        # init model
        self.build_model()
        # model_path = params.get("path_to_model", None)
        # if model_path is not None:
        #     self.load_model(model_path)

        if params.get("resume_training", False):
            model_path = os.path.join(
                params["output_path"], params["training_state_dir"], "pytorch_model.bin"
            )
        else:
            model_path = params.get("path_to_model", None)

        if model_path is not None:
            self.load_model(model_path, params["no_cuda"])

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict, strict=False)

    def build_model(self):
        self.model = BiEncoderModule(self.params)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )

    def encode_context(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        embedding_context, _ = self.model(
            token_idx_cands, segment_idx_cands, mask_cands, None, None, None
        )
        return embedding_context.cpu().detach()

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach()
        # TODO: why do we need cpu here?
        # return embedding_cands

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    def score_candidate(
            self,
            text_vecs,
            cand_vecs,
            random_negs=True,
            cand_encs=None,  # pre-computed candidate encoding.
    ):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t())

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        if random_negs:
            # train on random negatives
            return embedding_ctxt.mm(embedding_cands.t())
        else:
            # train on hard negatives
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
            scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
            scores = torch.squeeze(scores)
            return scores

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(self, context_input, cand_input, label_input=None):
        if self.params["hard_negatives_file"] is not None:
            cand_input = cand_input.reshape(-1, cand_input.shape[-1])

        flag = label_input is None
        scores = self.score_candidate(context_input, cand_input, flag)
        bs = scores.size(0)

        if label_input is None:
            if self.params["hard_negatives_file"] is not None:
                target = torch.LongTensor(
                    torch.arange(
                        0,
                        bs * (self.params["max_num_negatives"] + 1),
                        self.params["max_num_negatives"] + 1,
                    )
                )
            else:
                target = torch.LongTensor(torch.arange(bs))

            target = target.to(self.device)
            loss = F.cross_entropy(scores, target, reduction="mean")
        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            # TODO: add parameters?
            loss = loss_fct(scores, label_input)

        return loss, scores


class TypedBiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(TypedBiEncoderModule, self).__init__()
        ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        cand_bert = BertModel.from_pretrained(params["bert_model"])
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = ctxt_bert.config

        if params["type_embeddings_path"] == "":
            self.additional_type_embeddings = nn.Embedding(
                num_embeddings=params["num_types"],
                embedding_dim=params["type_embedding_dim"],
            )
        else:
            type_embedding_weights = torch.load(params["type_embeddings_path"])
            self.additional_type_embeddings = nn.Embedding.from_pretrained(
                embeddings=type_embedding_weights,
                freeze=params["freeze_type_embeddings"],
            )

        self.additional_up_project_linear = None
        if not params["no_linear_after_type_embeddings"]:
            self.additional_up_project_linear = nn.Linear(
                in_features=params["type_embedding_dim"],
                out_features=self.config.hidden_size,
                bias=True,
            )

        self.params = params

    def forward(
            self,
            token_idx_ctxt,
            segment_idx_ctxt,
            mask_ctxt,
            token_idx_cands,
            segment_idx_cands,
            mask_cands,
            types=None,
    ):
        """
        Calling this function with token_idx_ctxt will also return the type vectors.
        Isn't this ugly code? Yes it is. 
        """
        type_vectors_scaled_up = None
        embedding_ctxt = None

        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
            if types is not None:
                type_vectors = self.additional_type_embeddings(types)
                if self.additional_up_project_linear is not None:
                    type_vectors_scaled_up = self.additional_up_project_linear(
                        type_vectors
                    )
                else:
                    type_vectors_scaled_up = type_vectors

        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )

        if types is not None:
            return embedding_ctxt, embedding_cands, type_vectors_scaled_up
        else:
            return (
                embedding_ctxt,
                embedding_cands,
            )


class TypedBiEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(TypedBiEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        # init model
        self.build_model()
        # model_path = params.get("path_to_model", None)
        # if model_path is not None:
        #     self.load_model(model_path)

        if params.get("resume_training", False):
            model_path = os.path.join(
                params["output_path"], params["training_state_dir"], "pytorch_model.bin"
            )
        else:
            model_path = params.get("path_to_model", None)

        if model_path is not None:
            self.load_model(model_path, params["no_cuda"])

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        is_training = self.model.training

        self.summary_writer = None

    def set_summary_writer(self, summary_writer):
        self.summary_writer = summary_writer

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict, strict=False)

    def build_model(self):
        self.model = TypedBiEncoderModule(self.params)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )

    def encode_context(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        embedding_context, _ = self.model(
            token_idx_cands, segment_idx_cands, mask_cands, None, None, None
        )
        return embedding_context.cpu().detach()

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach()
        # TODO: why do we need cpu here?
        # return embedding_cands

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    # This method is for inference only!
    def score_candidate(
            self,
            text_vecs,
            cand_vecs,
            random_negs=True,
            cand_encs=None,  # pre-computed candidate encoding.
    ):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t())

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        if random_negs:
            # train on random negatives
            return embedding_ctxt.mm(embedding_cands.t())
        else:
            # train on hard negatives
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
            scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
            scores = torch.squeeze(scores)
            return scores

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward_inference(self, context_input, cand_input, label_input=None):
        flag = label_input is None
        scores = self.score_candidate(context_input, cand_input, flag)
        bs = scores.size(0)
        if label_input is None:
            target = torch.LongTensor(torch.arange(bs))
            target = target.to(self.device)
            loss = F.cross_entropy(scores, target, reduction="mean")
        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            # TODO: add parameters?
            loss = loss_fct(scores, label_input)
        return loss, scores

    def score_candidate_type_inference(
            self,
            text_vecs,
            cand_vecs,
            random_negs=True,
            cand_encs=None,  # pre-computed candidate encoding.
    ):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None,
        )

        type_vectors = self.model.module.type_embeddings.weight
        type_vectors = self.model.module.up_project_linear(type_vectors)
        type_prediction_scores = torch.matmul(
            embedding_ctxt, torch.transpose(type_vectors, 0, 1)
        )
        assert type_prediction_scores.shape[1] == self.params["num_types"]

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        embedding_ctxt = embedding_ctxt.to("cpu")
        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t()), type_prediction_scores

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )

        assert embedding_ctxt.shape[0] == text_vecs.shape[0]

        assert type_vectors.shape[0] == self.params["num_types"]

        # type_prediction_scores = torch.matmul(
        #     embedding_ctxt.unsqueeze(1), type_vectors.permute(0, 2, 1)
        # ).squeeze(1)

        if random_negs:
            # train on random negatives
            return embedding_ctxt.mm(embedding_cands.t()), type_prediction_scores
        else:
            raise NotImplementedError

            # train on hard negatives
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
            scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
            scores = torch.squeeze(scores)
            return scores

    def score_candidate_train(
            self,
            text_vecs,
            cand_vecs,
            type_vecs,
            random_negs=True,
            cand_encs=None,  # pre-computed candidate encoding.
    ):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _, type_vectors = self.model(
            token_idx_ctxt,
            segment_idx_ctxt,
            mask_ctxt,
            None,
            None,
            None,
            types=type_vecs,
        )

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t())

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )

        assert embedding_ctxt.shape[0] == text_vecs.shape[0]

        assert type_vectors.shape[0] == text_vecs.shape[0]
        assert type_vectors.shape[1] == self.params["max_type_list_len"]

        type_prediction_scores = torch.matmul(
            embedding_ctxt.unsqueeze(1), type_vectors.permute(0, 2, 1)
        ).squeeze(1)

        assert type_prediction_scores.shape[0] == text_vecs.shape[0]
        assert type_prediction_scores.shape[1] == self.params["max_type_list_len"]

        if random_negs:
            # train on random negatives
            return embedding_ctxt.mm(embedding_cands.t()), type_prediction_scores
        else:
            raise NotImplementedError

            # train on hard negatives
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
            scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
            scores = torch.squeeze(scores)
            return scores

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward_train(
            self,
            context_input,
            cand_input,
            type_input,
            type_labels,
            label_input=None,
            iteration_number=None,
    ):
        if self.params["hard_negatives_file"] is not None:
            cand_input = cand_input.reshape(-1, cand_input.shape[-1])

        flag = label_input is None
        # scores = self.score_candidate_train(context_input, cand_input, flag)
        scores, type_scores = self.score_candidate_train(
            text_vecs=context_input,
            cand_vecs=cand_input,
            type_vecs=type_input,
            random_negs=True,
            cand_encs=None,
        )
        bs = scores.size(0)

        if label_input is None:
            if self.params["hard_negatives_file"] is not None:
                target = torch.LongTensor(
                    torch.arange(
                        0,
                        bs * (self.params["max_num_negatives"] + 1),
                        self.params["max_num_negatives"] + 1,
                    )
                )
            else:
                target = torch.LongTensor(torch.arange(bs))

            target = target.to(self.device)
            entity_loss = F.cross_entropy(scores, target, reduction="mean")

            bce_loss_function = nn.BCEWithLogitsLoss(reduction="mean")
            type_loss = bce_loss_function(type_scores, type_labels)

            loss = (
                    self.params["blink_loss_weight"] * entity_loss
                    + self.params["type_loss_weight"] * type_loss
            )

            if self.params["tb"]:
                self.summary_writer.add_scalars(
                    main_tag="losses",
                    tag_scalar_dict={
                        "entity_loss": entity_loss.item(),
                        "type_loss": type_loss.item(),
                        "total_loss": loss.item(),
                    },
                    global_step=iteration_number,
                )
        else:
            raise NotImplementedError
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            # TODO: add parameters?
            loss = loss_fct(scores, label_input)

        return loss, scores

    def forward_type_inference(
            self, context_input, cand_input, type_labels, cand_encs, label_input=None,
    ):

        flag = label_input is None
        # scores = self.score_candidate_train(context_input, cand_input, flag)
        scores, type_scores = self.score_candidate_type_inference(
            text_vecs=context_input,
            cand_vecs=cand_input,
            random_negs=True,
            cand_encs=cand_encs,
        )

        return scores, type_scores
        bs = scores.size(0)

        if label_input is None:

            target = torch.LongTensor(torch.arange(bs))

            target = target.to(self.device)
            entity_loss = F.cross_entropy(scores, target, reduction="mean")

            bce_loss_function = nn.BCEWithLogitsLoss(reduction="mean")
            type_loss = bce_loss_function(type_scores, type_labels.to(torch.float))

            loss = (
                    self.params["blink_loss_weight"] * entity_loss
                    + self.params["type_loss_weight"] * type_loss
            )

            if self.params["tb"]:
                self.summary_writer.add_scalars(
                    main_tag="losses",
                    tag_scalar_dict={
                        "entity_loss": entity_loss.item(),
                        "type_loss": type_loss.item(),
                        "total_loss": loss.item(),
                    },
                    global_step=iteration_number,
                )
        else:
            raise NotImplementedError
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            # TODO: add parameters?
            loss = loss_fct(scores, label_input)

        return loss, scores, type_scores

    def forward(
            self,
            context_input,
            cand_input,
            type_input=None,
            type_labels=None,
            label_input=None,
            iteration_number=None,
    ):
        assert ((type_input is not None) and (type_labels is not None)) or (
                (type_input is None) and (type_labels is None)
        )

        if (type_input is not None) and (type_labels is not None):
            loss, scores = self.forward_train(
                context_input,
                cand_input,
                type_input,
                type_labels,
                label_input,
                iteration_number=iteration_number,
            )
        else:
            loss, scores = self.forward_inference(
                context_input, cand_input, label_input
            )

        # returning scores so that BLINK inference code doesn't break
        return loss, scores


class TypedBiEncoderModule2(torch.nn.Module):
    def __init__(self, params):
        super(TypedBiEncoderModule2, self).__init__()
        ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        cand_bert = BertModel.from_pretrained(params["bert_model"])
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = ctxt_bert.config

        if params["type_embeddings_path"] == "":
            self.additional_type_embeddings = nn.Embedding(
                num_embeddings=params["num_types"],
                embedding_dim=params["type_embedding_dim"],
            )
        else:
            type_embedding_weights = torch.load(params["type_embeddings_path"])
            self.additional_type_embeddings = nn.Embedding.from_pretrained(
                embeddings=type_embedding_weights,
                freeze=params["freeze_type_embeddings"],
            )

        self.additional_up_project_linear = None
        if not params["no_linear_after_type_embeddings"]:
            # self.additional_up_project_linear = nn.Linear(
            #     in_features=params["type_embedding_dim"],
            #     out_features=self.config.hidden_size,
            #     bias=True,
            # )

            self.additional_up_project_linear = nn.Sequential(
                nn.Linear(
                    in_features=params["type_embedding_dim"],
                    out_features=int(self.config.hidden_size / 3),
                    bias=True,
                ),
                nn.GELU(),
                nn.Linear(
                    in_features=int(self.config.hidden_size / 3),
                    out_features=self.config.hidden_size,
                    bias=True,
                ),
            )

            # self.additional_up_project_linear = nn.Sequential(
            #     nn.Linear(
            #         in_features=params["type_embedding_dim"],
            #         out_features=self.config.hidden_size,
            #         bias=True,
            #     ),
            #     nn.GELU(),
            #     nn.LayerNorm(self.config.hidden_size),
            # )

        self.params = params

    def get_type_vectors(self):
        type_vectors = self.additional_type_embeddings.weight
        if self.additional_up_project_linear is not None:
            type_vectors = self.additional_up_project_linear(type_vectors)
        return type_vectors

    def forward(
            self,
            token_idx_ctxt,
            segment_idx_ctxt,
            mask_ctxt,
            token_idx_cands,
            segment_idx_cands,
            mask_cands,
    ):

        embedding_ctxt = None
        if token_idx_ctxt is not None:
            if self.params["freeze_context_bert"]:
                with torch.no_grad():
                    embedding_ctxt = self.context_encoder(
                        token_idx_ctxt, segment_idx_ctxt, mask_ctxt
                    )
            else:
                embedding_ctxt = self.context_encoder(
                    token_idx_ctxt, segment_idx_ctxt, mask_ctxt
                )

        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )

        return embedding_ctxt, embedding_cands


class TypedBiEncoderRanker2(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(TypedBiEncoderRanker2, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        # init model
        self.build_model()
        # model_path = params.get("path_to_model", None)
        # if model_path is not None:
        #     self.load_model(model_path)

        if params.get("resume_training", False):
            model_path = os.path.join(
                params["output_path"], params["training_state_dir"], "pytorch_model.bin"
            )
        else:
            model_path = params.get("path_to_model", None)

        if model_path is not None:
            self.load_model(model_path, params["no_cuda"])

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.summary_writer = None

    def set_summary_writer(self, summary_writer):
        self.summary_writer = summary_writer

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict, strict=False)

    def build_model(self):
        self.model = TypedBiEncoderModule2(self.params)

    # during training, call this function each time before you use the type vectors
    def get_type_vectors(self):
        if self.data_parallel:
            return self.model.module.get_type_vectors()
        else:
            return self.model.get_type_vectors()

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self):
        assert False, "Dont call this"
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )

    def encode_context(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        embedding_context, _ = self.model(
            token_idx_cands, segment_idx_cands, mask_cands, None, None, None
        )
        return embedding_context.cpu().detach()

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach()
        # TODO: why do we need cpu here?
        # return embedding_cands

    def score_candidate(
            self,
            text_vecs,
            cand_vecs,
            random_negs=True,
            cand_encs=None,  # pre-computed candidate encoding.
            training_progress=1.0
    ):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None,
        )

        # this block will be hit during inference. Sometimes, the type_task_importance_scheduling arg might be missing in
        # self.params
        if self.params.get("type_task_importance_scheduling", "") == "grad_throttle":
            scale = get_aux_task_weight(training_progress, gamma=10)
            embedding_ctxt_for_type_pred = grad_throttle(x=embedding_ctxt, scale=scale)
        else:
            embedding_ctxt_for_type_pred = embedding_ctxt

        type_vectors = self.get_type_vectors()

        type_prediction_scores = embedding_ctxt_for_type_pred.mm(type_vectors.t())

        # Candidate encodings are given, do not re-compute
        # inference time
        if cand_encs is not None:
            entity_prediction_scores = embedding_ctxt.mm(cand_encs.t())
            return entity_prediction_scores, type_prediction_scores

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )

        assert embedding_ctxt.shape[0] == text_vecs.shape[0]

        entity_prediction_scores = embedding_ctxt.mm(embedding_cands.t())

        random_negs = True
        if random_negs:
            # train on random negatives
            return entity_prediction_scores, type_prediction_scores
        else:
            raise NotImplementedError

    def score_candidate_type_inference(
            self,
            text_vecs,
            cand_vecs,
            random_negs=True,
            cand_encs=None,  # pre-computed candidate encoding.
    ):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None,
        )

        # type_vectors= self.model.module.type_embeddings.weight
        # type_vectors = self.model.module.up_project_linear(type_vectors)
        type_vectors = self.get_type_vectors()

        # type_prediction_scores = torch.matmul(
        #     embedding_ctxt, torch.transpose(type_vectors, 0, 1)
        # )
        type_prediction_scores = embedding_ctxt.mm(type_vectors.t())

        type_prediction_scores = torch.matmul(
            embedding_ctxt, torch.transpose(type_vectors, 0, 1)
        )
        assert type_prediction_scores.shape[1] == self.params["num_types"]

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        embedding_ctxt = embedding_ctxt.to("cpu")
        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t()), type_prediction_scores

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )

        assert embedding_ctxt.shape[0] == text_vecs.shape[0]

        assert type_vectors.shape[0] == self.params["num_types"]

        # type_prediction_scores = torch.matmul(
        #     embedding_ctxt.unsqueeze(1), type_vectors.permute(0, 2, 1)
        # ).squeeze(1)

        if random_negs:
            # train on random negatives
            return embedding_ctxt.mm(embedding_cands.t()), type_prediction_scores
        else:
            raise NotImplementedError

            # train on hard negatives
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
            scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
            scores = torch.squeeze(scores)
            return scores

    def forward(
            self, context_input, cand_input, type_labels=None, iteration_number=None, training_progress=1.0
    ):
        if self.params["hard_negatives_file"] is not None:
            cand_input = cand_input.reshape(-1, cand_input.shape[-1])

        entity_scores, type_scores = self.score_candidate(
            text_vecs=context_input,
            cand_vecs=cand_input,
            random_negs=True,
            cand_encs=None,
            training_progress=training_progress
        )
        bs = entity_scores.size(0)

        loss = None

        if type_labels is not None:
            if self.params["hard_negatives_file"] is not None:
                target = torch.LongTensor(
                    torch.arange(
                        0,
                        bs * (self.params["max_num_negatives"] + 1),
                        self.params["max_num_negatives"] + 1,
                    )
                )
            else:
                target = torch.LongTensor(torch.arange(bs))

            target = target.to(self.device)
            entity_loss = F.cross_entropy(entity_scores, target, reduction="mean")

            bce_loss_function = nn.BCEWithLogitsLoss(reduction="none")

            type_loss_unreduced = bce_loss_function(type_scores, type_labels)

            # average loss for all positive types in the batch
            type_loss_positives = safe_divide(
                (type_loss_unreduced * type_labels).sum(), type_labels.sum()
            )

            # average loss for all negative types in the batch
            type_loss_negatives = safe_divide(
                (type_loss_unreduced * (1 - type_labels)).sum(), (1 - type_labels).sum()
            )

            type_loss = (
                                self.params["type_loss_weight_positive"] * type_loss_positives
                        ) + (self.params["type_loss_weight_negative"] * type_loss_negatives)

            if self.params["type_task_importance_scheduling"] == "loss_weight":
                type_loss_weight = get_aux_task_weight(training_progress, gamma=10)
            else:
                type_loss_weight = self.params["type_loss_weight"]

            loss = (
                    self.params["blink_loss_weight"] * entity_loss
                    + type_loss_weight * type_loss
            )

            type_probs = torch.sigmoid(type_scores)

            if self.params["tb"]:
                self.summary_writer.add_scalars(
                    main_tag="losses/main",
                    tag_scalar_dict={
                        "entity_loss": entity_loss.item(),
                        "type_loss": type_loss.item(),
                        "total_loss": loss.item(),
                    },
                    global_step=iteration_number,
                )

                self.summary_writer.add_scalars(
                    main_tag="losses/types",
                    tag_scalar_dict={
                        "positive": type_loss_positives.item(),
                        "negative": type_loss_negatives.item(),
                    },
                    global_step=iteration_number,
                )

                average_positive_type_probability = safe_divide(
                    (type_probs * type_labels).sum().item(), type_labels.sum().item()
                )

                average_negative_type_probability = safe_divide(
                    (type_probs * (1 - type_labels)).sum().item(),
                    (1 - type_labels).sum().item(),
                )

                self.summary_writer.add_scalars(
                    main_tag="Average_type_probability",
                    tag_scalar_dict={
                        "positive_types": average_positive_type_probability,
                        "negative_types": average_negative_type_probability,
                    },
                    global_step=iteration_number,
                )

        all_scores = {"entity_scores": entity_scores, "type_probs": type_probs}

        return loss, all_scores

    def forward_inference(
            self, context_input, cand_input, cand_encs
    ):

        entity_scores, type_scores = self.score_candidate(
            text_vecs=context_input,
            cand_vecs=cand_input,
            random_negs=True,
            cand_encs=cand_encs,
            training_progress=1.0
        )

        type_probabilities = torch.sigmoid(type_scores)

        return entity_scores, type_probabilities


class TypedBiEncoderModule3(torch.nn.Module):
    def __init__(self, params):
        super(TypedBiEncoderModule3, self).__init__()
        ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        cand_bert = BertModel.from_pretrained(params["bert_model"])
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = ctxt_bert.config

        if params["type_embeddings_path"] == "":
            self.additional_type_embeddings = nn.Embedding(
                num_embeddings=params["num_types"],
                embedding_dim=params["type_embedding_dim"],
            )
        else:
            type_embedding_weights = torch.load(params["type_embeddings_path"])
            self.additional_type_embeddings = nn.Embedding.from_pretrained(
                embeddings=type_embedding_weights,
                freeze=params["freeze_type_embeddings"],
            )

        self.additional_up_project_linear = None
        if not params["no_linear_after_type_embeddings"]:
            # self.additional_up_project_linear = nn.Linear(
            #     in_features=params["type_embedding_dim"],
            #     out_features=self.config.hidden_size,
            #     bias=True,
            # )

            self.additional_up_project_linear = nn.Sequential(
                nn.Linear(
                    in_features=params["type_embedding_dim"],
                    out_features=int(self.config.hidden_size / 3),
                    bias=True,
                ),
                nn.GELU(),
                nn.Linear(
                    in_features=int(self.config.hidden_size / 3),
                    out_features=self.config.hidden_size,
                    bias=True,
                ),
            )

            # self.additional_up_project_linear = nn.Sequential(
            #     nn.Linear(
            #         in_features=params["type_embedding_dim"],
            #         out_features=self.config.hidden_size,
            #         bias=True,
            #     ),
            #     nn.GELU(),
            #     nn.LayerNorm(self.config.hidden_size),
            # )

        self.additional_ontology_nn = OntologyNN(params)
        self.params = params

    def get_type_vectors(self):
        type_vectors = self.additional_type_embeddings.weight
        if self.additional_up_project_linear is not None:
            type_vectors = self.additional_up_project_linear(type_vectors)
        return type_vectors

    def forward(
            self,
            token_idx_ctxt,
            segment_idx_ctxt,
            mask_ctxt,
            token_idx_cands,
            segment_idx_cands,
            mask_cands,
    ):

        embedding_ctxt = None
        if token_idx_ctxt is not None:
            if self.params["freeze_context_bert"]:
                with torch.no_grad():
                    embedding_ctxt = self.context_encoder(
                        token_idx_ctxt, segment_idx_ctxt, mask_ctxt
                    )
            else:
                embedding_ctxt = self.context_encoder(
                    token_idx_ctxt, segment_idx_ctxt, mask_ctxt
                )

        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )

        return embedding_ctxt, embedding_cands


class TypedBiEncoderRanker3(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(TypedBiEncoderRanker3, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        # init model
        self.build_model()
        # model_path = params.get("path_to_model", None)
        # if model_path is not None:
        #     self.load_model(model_path)

        if params.get("resume_training", False):
            model_path = os.path.join(
                params["output_path"], params["training_state_dir"], "pytorch_model.bin"
            )
        else:
            model_path = params.get("path_to_model", None)

        if model_path is not None:
            self.load_model(model_path, params["no_cuda"])

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.summary_writer = None

    def set_summary_writer(self, summary_writer):
        self.summary_writer = summary_writer

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict, strict=False)

    def build_model(self):
        self.model = TypedBiEncoderModule3(self.params)

    # during training, call this function each time before you use the type vectors
    def get_type_vectors(self):
        if self.data_parallel:
            return self.model.module.get_type_vectors()
        else:
            return self.model.get_type_vectors()

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )

    def encode_context(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        embedding_context, _ = self.model(
            token_idx_cands, segment_idx_cands, mask_cands, None, None, None
        )
        return embedding_context.cpu().detach()

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach()
        # TODO: why do we need cpu here?
        # return embedding_cands

    def score_candidate(
            self,
            text_vecs,
            cand_vecs,
            random_negs=True,
            cand_encs=None,  # pre-computed candidate encoding.
    ):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None,
        )

        type_vectors = self.get_type_vectors()
        type_prediction_scores = embedding_ctxt.mm(type_vectors.t())

        # Candidate encodings are given, do not re-compute
        # inference time
        if cand_encs is not None:
            entity_prediction_scores = embedding_ctxt.mm(cand_encs.t())
            return entity_prediction_scores, type_prediction_scores

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )

        assert embedding_ctxt.shape[0] == text_vecs.shape[0]

        entity_prediction_scores = embedding_ctxt.mm(embedding_cands.t())

        random_negs = True
        if random_negs:
            # train on random negatives
            return entity_prediction_scores, type_prediction_scores
        else:
            raise NotImplementedError

    def score_candidate_type_inference(
            self,
            text_vecs,
            cand_vecs,
            random_negs=True,
            cand_encs=None,  # pre-computed candidate encoding.
    ):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None,
        )

        # type_vectors= self.model.module.type_embeddings.weight
        # type_vectors = self.model.module.up_project_linear(type_vectors)
        type_vectors = self.get_type_vectors()

        # type_prediction_scores = torch.matmul(
        #     embedding_ctxt, torch.transpose(type_vectors, 0, 1)
        # )
        type_prediction_scores = embedding_ctxt.mm(type_vectors.t())

        type_prediction_scores = torch.matmul(
            embedding_ctxt, torch.transpose(type_vectors, 0, 1)
        )
        assert type_prediction_scores.shape[1] == self.params["num_types"]

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        embedding_ctxt = embedding_ctxt.to("cpu")
        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t()), type_prediction_scores

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )

        assert embedding_ctxt.shape[0] == text_vecs.shape[0]

        assert type_vectors.shape[0] == self.params["num_types"]

        # type_prediction_scores = torch.matmul(
        #     embedding_ctxt.unsqueeze(1), type_vectors.permute(0, 2, 1)
        # ).squeeze(1)

        if random_negs:
            # train on random negatives
            return embedding_ctxt.mm(embedding_cands.t()), type_prediction_scores
        else:
            raise NotImplementedError

    def forward(
            self, context_input, cand_input, type_labels=None, iteration_number=None, training_progress=1.0
    ):
        """
        type_labels is overloaded. it contains the type labels (type_labels[:, 0, :]) and the descendant mask
        (type_labels[:, 1, :]).
        """

        if self.params["hard_negatives_file"] is not None:
            cand_input = cand_input.reshape(-1, cand_input.shape[-1])

        entity_scores, type_scores = self.score_candidate(
            text_vecs=context_input,
            cand_vecs=cand_input,
            random_negs=True,
            cand_encs=None,
        )

        type_probabilities = torch.sigmoid(type_scores)

        if self.params["data_parallel"]:
            revised_type_probabilities = self.model.module.additional_ontology_nn(type_probabilities)
        else:
            revised_type_probabilities = self.model.additional_ontology_nn(type_probabilities)

        bs = entity_scores.size(0)

        loss = None

        if type_labels is not None:
            type_labels_and_descendant_mask = type_labels

            assert len(type_labels_and_descendant_mask.shape) == 3
            assert type_labels_and_descendant_mask.shape[1] == 2
            assert type_labels_and_descendant_mask.shape[0] == context_input.shape[0]

            type_labels = type_labels_and_descendant_mask[:, 0, :]
            descendant_mask = type_labels_and_descendant_mask[:, 1, :]

            assert descendant_mask.shape == type_labels.shape
            assert descendant_mask.shape[0] == type_labels.shape[0] == context_input.shape[0]

            if self.params["hard_negatives_file"] is not None:
                target = torch.LongTensor(torch.arange(0, bs * (self.params["max_num_negatives"] + 1),
                                                       self.params["max_num_negatives"] + 1, ))
            else:
                target = torch.LongTensor(torch.arange(bs))

            target = target.to(self.device)
            entity_loss = F.cross_entropy(entity_scores, target, reduction="mean")

            bce_loss_function = nn.BCELoss(reduction="none")

            type_loss_unreduced = bce_loss_function(revised_type_probabilities, type_labels)

            # 0 loss for descendants of the finest gold types of the entity
            type_loss_unreduced = type_loss_unreduced * (1 - descendant_mask)

            # average loss for all positive types in the batch
            type_loss_positives = safe_divide(
                (type_loss_unreduced * type_labels).sum(), type_labels.sum()
            )

            # average loss for all negative types in the batch
            type_loss_negatives = safe_divide(
                (type_loss_unreduced * (1 - type_labels)).sum(), (1 - type_labels).sum()
            )

            type_loss = (
                                self.params["type_loss_weight_positive"] * type_loss_positives
                        ) + (self.params["type_loss_weight_negative"] * type_loss_negatives)

            loss = (
                    self.params["blink_loss_weight"] * entity_loss
                    + self.params["type_loss_weight"] * type_loss
            )

            if self.params["tb"]:
                self.summary_writer.add_scalars(
                    main_tag="losses/main",
                    tag_scalar_dict={
                        "entity_loss": entity_loss.item(),
                        "type_loss": type_loss.item(),
                        "total_loss": loss.item(),
                    },
                    global_step=iteration_number,
                )

                self.summary_writer.add_scalars(
                    main_tag="losses/types",
                    tag_scalar_dict={
                        "positive": type_loss_positives.item(),
                        "negative": type_loss_negatives.item(),
                    },
                    global_step=iteration_number,
                )

                average_positive_type_probability = safe_divide(
                    (revised_type_probabilities * type_labels).sum().item(), type_labels.sum().item()
                )

                average_negative_type_probability = safe_divide(
                    (revised_type_probabilities * (1 - type_labels)).sum().item(),
                    (1 - type_labels).sum().item(),
                )

                self.summary_writer.add_scalars(
                    main_tag="Average_type_probability",
                    tag_scalar_dict={
                        "positive_types": average_positive_type_probability,
                        "negative_types": average_negative_type_probability,
                    },
                    global_step=iteration_number,
                )

        all_scores = {"entity_scores": entity_scores, "type_probs": revised_type_probabilities}

        return loss, all_scores

    def forward_inference(
            self, context_input, cand_input, cand_encs,
    ):
        entity_scores, type_scores = self.score_candidate(
            text_vecs=context_input,
            cand_vecs=cand_input,
            random_negs=True,
            cand_encs=cand_encs,
        )

        type_probabilities = torch.sigmoid(type_scores)

        if self.params["data_parallel"]:
            revised_type_probabilities = self.model.module.additional_ontology_nn(type_probabilities)
        else:
            revised_type_probabilities = self.model.additional_ontology_nn(type_probabilities)

        return entity_scores, revised_type_probabilities


class TypedBiEncoderModule4(torch.nn.Module):
    def __init__(self, params):
        super(TypedBiEncoderModule4, self).__init__()
        ctxt_bert = BertModel.from_pretrained(params["bert_model"], output_hidden_states=True)
        cand_bert = BertModel.from_pretrained(params["bert_model"])
        self.context_encoder = BertEncoder2(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
            cls_start=params["cls_start"],
            cls_end=params["cls_end"]
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = ctxt_bert.config

        if params["type_embeddings_path"] == "":
            self.additional_type_embeddings = nn.Embedding(
                num_embeddings=params["num_types"],
                embedding_dim=params["type_embedding_dim"],
            )
        else:
            type_embedding_weights = torch.load(params["type_embeddings_path"])
            self.additional_type_embeddings = nn.Embedding.from_pretrained(
                embeddings=type_embedding_weights,
                freeze=params["freeze_type_embeddings"],
            )

        self.additional_up_project_linear = None
        if not params["no_linear_after_type_embeddings"]:
            # self.additional_up_project_linear = nn.Linear(
            #     in_features=params["type_embedding_dim"],
            #     out_features=self.config.hidden_size,
            #     bias=True,
            # )

            self.additional_up_project_linear = nn.Sequential(
                nn.Linear(
                    in_features=params["type_embedding_dim"],
                    out_features=int(self.config.hidden_size / 3),
                    bias=True,
                ),
                nn.GELU(),
                nn.Linear(
                    in_features=int(self.config.hidden_size / 3),
                    out_features=self.config.hidden_size,
                    bias=True,
                ),
            )

            # self.additional_up_project_linear = nn.Sequential(
            #     nn.Linear(
            #         in_features=params["type_embedding_dim"],
            #         out_features=self.config.hidden_size,
            #         bias=True,
            #     ),
            #     nn.GELU(),
            #     nn.LayerNorm(self.config.hidden_size),
            # )

        # self.additional_ontology_nn = OntologyNN(params)
        self.params = params

    def get_type_vectors(self):
        type_vectors = self.additional_type_embeddings.weight
        if self.additional_up_project_linear is not None:
            type_vectors = self.additional_up_project_linear(type_vectors)
        return type_vectors

    def forward_context(
            self,
            token_idx_ctxt,
            segment_idx_ctxt,
            mask_ctxt,
    ):
        embedding_ctxt = self.context_encoder(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt
        )

        return embedding_ctxt

    def forward_candidate(
            self,
            token_idx_cands,
            segment_idx_cands,
            mask_cands,
    ):

        embedding_cands = self.cand_encoder(
            token_idx_cands, segment_idx_cands, mask_cands
        )

        return embedding_cands

    def forward(
            self,
            token_idx_ctxt,
            segment_idx_ctxt,
            mask_ctxt,
            token_idx_cands,
            segment_idx_cands,
            mask_cands,
    ):

        if token_idx_ctxt is not None:
            return self.forward_context(
                token_idx_ctxt,
                segment_idx_ctxt,
                mask_ctxt,
            )
        else:
            return self.forward_candidate(
                token_idx_cands,
                segment_idx_cands,
                mask_cands,
            )


class TypedBiEncoderRanker4(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(TypedBiEncoderRanker4, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )

        # init model
        self.build_model()

        if params.get("resume_training", False):
            model_path = os.path.join(
                params["output_path"], params["training_state_dir"], "pytorch_model.bin"
            )
        else:
            model_path = params.get("path_to_model", None)

        if model_path is not None:
            self.load_model(model_path, params["no_cuda"])

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.summary_writer = None

    def set_summary_writer(self, summary_writer):
        self.summary_writer = summary_writer

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict, strict=False)

    def build_model(self):
        self.model = TypedBiEncoderModule4(self.params)

    # during training, call this function each time before you use the type vectors
    def get_type_vectors(self):
        if self.data_parallel:
            return self.model.module.get_type_vectors()
        else:
            return self.model.get_type_vectors()

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self):
        assert False, "Dont call this"
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )

    def encode_context(self, cands):
        # encodes context for the purpose of entity linking and NOT type prediction.

        text_vecs = cands

        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )
        return embedding_ctxt["for_entity_prediction"].cpu().detach()

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach()

    def score_candidate(
            self,
            text_vecs,
            cand_vecs,
            random_negs=True,
            cand_encs=None,  # pre-computed candidate encoding.
            training_progress=1.0
    ):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )

        if self.params.get("type_task_importance_scheduling", "") == "grad_throttle":
            scale = get_aux_task_weight(training_progress, gamma=10)
            embedding_ctxt["for_type_prediction"] = grad_throttle(x=embedding_ctxt["for_type_prediction"], scale=scale)

        type_vectors = self.get_type_vectors()

        type_prediction_scores = embedding_ctxt["for_type_prediction"].mm(type_vectors.t())

        # Candidate encodings are given, do not re-compute
        # inference time
        if cand_encs is not None:
            entity_prediction_scores = embedding_ctxt["for_entity_prediction"].mm(cand_encs.t())
            return entity_prediction_scores, type_prediction_scores

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )

        assert embedding_ctxt["for_entity_prediction"].shape[0] == text_vecs.shape[0]

        entity_prediction_scores = embedding_ctxt["for_entity_prediction"].mm(embedding_cands.t())

        random_negs = True
        if random_negs:
            # train on random negatives
            return entity_prediction_scores, type_prediction_scores
        else:
            raise NotImplementedError

    def score_candidate_type_inference(
            self,
            text_vecs,
            cand_vecs,
            random_negs=True,
            cand_encs=None,  # pre-computed candidate encoding.
    ):
        raise NotImplementedError

    def forward(
            self, context_input, cand_input, type_labels=None, iteration_number=None, training_progress=1.0
    ):
        if self.params["hard_negatives_file"] is not None:
            cand_input = cand_input.reshape(-1, cand_input.shape[-1])

        entity_scores, type_scores = self.score_candidate(
            text_vecs=context_input,
            cand_vecs=cand_input,
            random_negs=True,
            cand_encs=None,
            training_progress=training_progress
        )
        bs = entity_scores.size(0)

        loss = None

        if type_labels is not None:
            if self.params["hard_negatives_file"] is not None:
                target = torch.LongTensor(
                    torch.arange(
                        0,
                        bs * (self.params["max_num_negatives"] + 1),
                        self.params["max_num_negatives"] + 1,
                    )
                )
            else:
                target = torch.LongTensor(torch.arange(bs))

            target = target.to(self.device)
            entity_loss = F.cross_entropy(entity_scores, target, reduction="mean")

            bce_loss_function = nn.BCEWithLogitsLoss(reduction="none")

            type_loss_unreduced = bce_loss_function(type_scores, type_labels)

            # average loss for all positive types in the batch
            type_loss_positives = safe_divide(
                (type_loss_unreduced * type_labels).sum(), type_labels.sum()
            )

            # average loss for all negative types in the batch
            type_loss_negatives = safe_divide(
                (type_loss_unreduced * (1 - type_labels)).sum(), (1 - type_labels).sum()
            )

            type_loss = (
                                self.params["type_loss_weight_positive"] * type_loss_positives
                        ) + (self.params["type_loss_weight_negative"] * type_loss_negatives)

            if self.params["type_task_importance_scheduling"] == "loss_weight":
                type_loss_weight = get_aux_task_weight(training_progress, gamma=10)
            else:
                type_loss_weight = self.params["type_loss_weight"]

            loss = (
                    self.params["blink_loss_weight"] * entity_loss
                    + type_loss_weight * type_loss
            )

            type_probs = torch.sigmoid(type_scores)

            if self.params["tb"]:
                self.summary_writer.add_scalars(
                    main_tag="losses/main",
                    tag_scalar_dict={
                        "entity_loss": entity_loss.item(),
                        "type_loss": type_loss.item(),
                        "total_loss": loss.item(),
                    },
                    global_step=iteration_number,
                )

                self.summary_writer.add_scalars(
                    main_tag="losses/types",
                    tag_scalar_dict={
                        "positive": type_loss_positives.item(),
                        "negative": type_loss_negatives.item(),
                    },
                    global_step=iteration_number,
                )

                average_positive_type_probability = safe_divide(
                    (type_probs * type_labels).sum().item(), type_labels.sum().item()
                )

                average_negative_type_probability = safe_divide(
                    (type_probs * (1 - type_labels)).sum().item(),
                    (1 - type_labels).sum().item(),
                )

                self.summary_writer.add_scalars(
                    main_tag="Average_type_probability",
                    tag_scalar_dict={
                        "positive_types": average_positive_type_probability,
                        "negative_types": average_negative_type_probability,
                    },
                    global_step=iteration_number,
                )

        all_scores = {"entity_scores": entity_scores, "type_probs": type_probs}

        return loss, all_scores

    def forward_inference(
            self, context_input, cand_input, cand_encs,
    ):
        entity_scores, type_scores = self.score_candidate(
            text_vecs=context_input,
            cand_vecs=cand_input,
            random_negs=True,
            cand_encs=cand_encs,
            training_progress=1.0
        )

        type_probabilities = torch.sigmoid(type_scores)

        return entity_scores, type_probabilities


class TypedBiEncoderModule5(torch.nn.Module):
    def __init__(self, params):
        super(TypedBiEncoderModule5, self).__init__()
        ctxt_bert = BertModel.from_pretrained(params["bert_model"], output_hidden_states=True)
        cand_bert = BertModel.from_pretrained(params["bert_model"])
        self.context_encoder = BertEncoder2(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = ctxt_bert.config

        if params["type_embeddings_path"] == "":
            self.additional_type_embeddings = nn.Embedding(
                num_embeddings=params["num_types"],
                embedding_dim=params["type_embedding_dim"],
            )
        else:
            type_embedding_weights = torch.load(params["type_embeddings_path"])
            self.additional_type_embeddings = nn.Embedding.from_pretrained(
                embeddings=type_embedding_weights,
                freeze=params["freeze_type_embeddings"],
            )

        self.additional_up_project_linear = None
        if not params["no_linear_after_type_embeddings"]:
            # self.additional_up_project_linear = nn.Linear(
            #     in_features=params["type_embedding_dim"],
            #     out_features=self.config.hidden_size,
            #     bias=True,
            # )

            self.additional_up_project_linear = nn.Sequential(
                nn.Linear(
                    in_features=params["type_embedding_dim"],
                    out_features=int(self.config.hidden_size / 3),
                    bias=True,
                ),
                nn.GELU(),
                nn.Linear(
                    in_features=int(self.config.hidden_size / 3),
                    out_features=self.config.hidden_size,
                    bias=True,
                ),
            )

            # self.additional_up_project_linear = nn.Sequential(
            #     nn.Linear(
            #         in_features=params["type_embedding_dim"],
            #         out_features=self.config.hidden_size,
            #         bias=True,
            #     ),
            #     nn.GELU(),
            #     nn.LayerNorm(self.config.hidden_size),
            # )

        self.additional_ontology_nn = OntologyNN(params)
        self.params = params

    def get_type_vectors(self):
        type_vectors = self.additional_type_embeddings.weight
        if self.additional_up_project_linear is not None:
            type_vectors = self.additional_up_project_linear(type_vectors)
        return type_vectors

    def forward_context(
            self,
            token_idx_ctxt,
            segment_idx_ctxt,
            mask_ctxt,
    ):
        embedding_ctxt = self.context_encoder(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt
        )

        return embedding_ctxt

    def forward_candidate(
            self,
            token_idx_cands,
            segment_idx_cands,
            mask_cands,
    ):

        embedding_cands = self.cand_encoder(
            token_idx_cands, segment_idx_cands, mask_cands
        )

        return embedding_cands

    def forward(
            self,
            token_idx_ctxt,
            segment_idx_ctxt,
            mask_ctxt,
            token_idx_cands,
            segment_idx_cands,
            mask_cands,
    ):
        if token_idx_ctxt is not None:
            return self.forward_context(
                token_idx_ctxt,
                segment_idx_ctxt,
                mask_ctxt,
            )
        else:
            return self.forward_candidate(
                token_idx_cands,
                segment_idx_cands,
                mask_cands,
            )


class TypedBiEncoderRanker5(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(TypedBiEncoderRanker5, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        # init model
        self.build_model()
        # model_path = params.get("path_to_model", None)
        # if model_path is not None:
        #     self.load_model(model_path)

        if params.get("resume_training", False):
            model_path = os.path.join(
                params["output_path"], params["training_state_dir"], "pytorch_model.bin"
            )
        else:
            model_path = params.get("path_to_model", None)

        if model_path is not None:
            self.load_model(model_path, params["no_cuda"])

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.summary_writer = None

    def set_summary_writer(self, summary_writer):
        self.summary_writer = summary_writer

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict, strict=False)

    def build_model(self):
        self.model = TypedBiEncoderModule5(self.params)

    # during training, call this function each time before you use the type vectors
    def get_type_vectors(self):
        if self.data_parallel:
            return self.model.module.get_type_vectors()
        else:
            return self.model.get_type_vectors()

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )

    def encode_context(self, cands):
        # encodes context for the purpose of entity linking and NOT type prediction.

        text_vecs = cands

        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )
        return embedding_ctxt["for_entity_prediction"].cpu().detach()

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach()

    def score_candidate(
            self,
            text_vecs,
            cand_vecs,
            random_negs=True,
            cand_encs=None,  # pre-computed candidate encoding.
            training_progress=1.0
    ):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None,
        )

        if self.params.get("type_task_importance_scheduling", "") == "grad_throttle":
            scale = get_aux_task_weight(training_progress, gamma=10)
            embedding_ctxt["for_type_prediction"] = grad_throttle(x=embedding_ctxt["for_type_prediction"], scale=scale)

        type_vectors = self.get_type_vectors()

        type_prediction_scores = embedding_ctxt["for_type_prediction"].mm(type_vectors.t())

        # Candidate encodings are given, do not re-compute
        # inference time
        if cand_encs is not None:
            entity_prediction_scores = embedding_ctxt["for_entity_prediction"].mm(cand_encs.t())
            return entity_prediction_scores, type_prediction_scores

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )

        assert embedding_ctxt["for_entity_prediction"].shape[0] == text_vecs.shape[0]

        entity_prediction_scores = embedding_ctxt["for_entity_prediction"].mm(embedding_cands.t())

        random_negs = True
        if random_negs:
            # train on random negatives
            return entity_prediction_scores, type_prediction_scores
        else:
            raise NotImplementedError

    def score_candidate_type_inference(
            self,
            text_vecs,
            cand_vecs,
            random_negs=True,
            cand_encs=None,  # pre-computed candidate encoding.
    ):
        raise NotImplementedError

    def forward(
            self, context_input, cand_input, type_labels=None, iteration_number=None, training_progress=1.0
    ):
        """
        type_labels is overloaded. it contains the type labels (type_labels[:, 0, :]) and the descendant mask
        (type_labels[:, 1, :]).
        """

        if self.params["hard_negatives_file"] is not None:
            cand_input = cand_input.reshape(-1, cand_input.shape[-1])

        entity_scores, type_scores = self.score_candidate(
            text_vecs=context_input,
            cand_vecs=cand_input,
            random_negs=True,
            cand_encs=None,
            training_progress=training_progress
        )

        type_probabilities = torch.sigmoid(type_scores)

        if self.params["data_parallel"]:
            revised_type_probabilities = self.model.module.additional_ontology_nn(type_probabilities)
        else:
            revised_type_probabilities = self.model.additional_ontology_nn(type_probabilities)

        bs = entity_scores.size(0)

        loss = None

        if type_labels is not None:
            type_labels_and_descendant_mask = type_labels

            assert len(type_labels_and_descendant_mask.shape) == 3
            assert type_labels_and_descendant_mask.shape[1] == 2
            assert type_labels_and_descendant_mask.shape[0] == context_input.shape[0]

            type_labels = type_labels_and_descendant_mask[:, 0, :]
            descendant_mask = type_labels_and_descendant_mask[:, 1, :]

            assert descendant_mask.shape == type_labels.shape
            assert descendant_mask.shape[0] == type_labels.shape[0] == context_input.shape[0]

            if self.params["hard_negatives_file"] is not None:
                target = torch.LongTensor(torch.arange(0, bs * (self.params["max_num_negatives"] + 1),
                                                       self.params["max_num_negatives"] + 1, ))
            else:
                target = torch.LongTensor(torch.arange(bs))

            target = target.to(self.device)
            entity_loss = F.cross_entropy(entity_scores, target, reduction="mean")

            bce_loss_function = nn.BCELoss(reduction="none")

            type_loss_unreduced = bce_loss_function(revised_type_probabilities, type_labels)

            # 0 loss for descendants of the finest gold types of the entity
            type_loss_unreduced = type_loss_unreduced * (1 - descendant_mask)

            # average loss for all positive types in the batch
            type_loss_positives = safe_divide(
                (type_loss_unreduced * type_labels).sum(), type_labels.sum()
            )

            # average loss for all negative types in the batch
            type_loss_negatives = safe_divide(
                (type_loss_unreduced * (1 - type_labels)).sum(), (1 - type_labels).sum()
            )

            type_loss = (
                                self.params["type_loss_weight_positive"] * type_loss_positives
                        ) + (self.params["type_loss_weight_negative"] * type_loss_negatives)

            if self.params["type_task_importance_scheduling"] == "loss_weight":
                type_loss_weight = get_aux_task_weight(training_progress, gamma=10)
            else:
                type_loss_weight = self.params["type_loss_weight"]

            loss = (
                    self.params["blink_loss_weight"] * entity_loss
                    + type_loss_weight * type_loss
            )

            if self.params["tb"]:
                self.summary_writer.add_scalars(
                    main_tag="losses/main",
                    tag_scalar_dict={
                        "entity_loss": entity_loss.item(),
                        "type_loss": type_loss.item(),
                        "total_loss": loss.item(),
                    },
                    global_step=iteration_number,
                )

                self.summary_writer.add_scalars(
                    main_tag="losses/types",
                    tag_scalar_dict={
                        "positive": type_loss_positives.item(),
                        "negative": type_loss_negatives.item(),
                    },
                    global_step=iteration_number,
                )

                average_positive_type_probability = safe_divide(
                    (revised_type_probabilities * type_labels).sum().item(), type_labels.sum().item()
                )

                average_negative_type_probability = safe_divide(
                    (revised_type_probabilities * (1 - type_labels)).sum().item(),
                    (1 - type_labels).sum().item(),
                )

                self.summary_writer.add_scalars(
                    main_tag="Average_type_probability",
                    tag_scalar_dict={
                        "positive_types": average_positive_type_probability,
                        "negative_types": average_negative_type_probability,
                    },
                    global_step=iteration_number,
                )

        all_scores = {"entity_scores": entity_scores, "type_probs": revised_type_probabilities}

        return loss, all_scores

    def forward_inference(
            self, context_input, cand_input, cand_encs,
    ):
        entity_scores, type_scores = self.score_candidate(
            text_vecs=context_input,
            cand_vecs=cand_input,
            random_negs=True,
            cand_encs=cand_encs,
            training_progress=1.0
        )

        type_probabilities = torch.sigmoid(type_scores)

        if self.params["data_parallel"]:
            revised_type_probabilities = self.model.module.additional_ontology_nn(type_probabilities)
        else:
            revised_type_probabilities = self.model.additional_ontology_nn(type_probabilities)

        return entity_scores, revised_type_probabilities


def get_type_model(model_number, params):
    if model_number == 1:
        # negative sampling of types
        reranker = TypedBiEncoderRanker(params)
        tokenizer = reranker.tokenizer
        model = reranker.model
    elif model_number == 2:
        # considers all types while computing type loss
        reranker = TypedBiEncoderRanker2(params)
        tokenizer = reranker.tokenizer
        model = reranker.model
    elif model_number == 3:
        # godel and lukasiewicz
        reranker = TypedBiEncoderRanker3(params)
        tokenizer = reranker.tokenizer
        model = reranker.model
    elif model_number == 4:
        # model_number == 2, but lower layer CLS vectors are used for type prediction
        reranker = TypedBiEncoderRanker4(params)
        tokenizer = reranker.tokenizer
        model = reranker.model
    elif model_number == 5:
        # model_number == 3 + lower layer CLS vectors are used for type prediction + loss scheduling, etc
        reranker = TypedBiEncoderRanker5(params)
        tokenizer = reranker.tokenizer
        model = reranker.model
    else:
        assert False, "Unsupported value given for type_model"

    return reranker, tokenizer, model
