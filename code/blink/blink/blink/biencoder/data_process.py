# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from typing import Type
import torch
from tqdm import tqdm, trange

import io

import json

import numpy as np

from torch.utils.data import DataLoader, TensorDataset

from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.biencoder.zeshel_utils import world_to_id
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG


def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


def get_context_representation(
        sample,
        tokenizer,
        max_seq_length,
        mention_key="mention",
        context_key="context",
        ent_start_token=ENT_START_TAG,
        ent_end_token=ENT_END_TAG,
):
    mention_tokens = []
    if sample[mention_key] and len(sample[mention_key]) > 0:
        mention_tokens = tokenizer.tokenize(sample[mention_key])

        # A few mentions are very long
        # making sure that len(mention_tokens) is never more than half the max_seq_length
        if len(mention_tokens) > ((max_seq_length // 2) - 4):
            mention_tokens = mention_tokens[: (max_seq_length // 2) - 4]

        mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]

    context_left = sample[context_key + "_left"]
    context_right = sample[context_key + "_right"]
    context_left = tokenizer.tokenize(context_left)
    context_right = tokenizer.tokenize(context_right)

    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = (
            context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
    )

    context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": context_tokens,
        "ids": input_ids,
    }


def get_candidate_representation(
        candidate_desc,
        tokenizer,
        max_seq_length,
        candidate_title=None,
        title_tag=ENT_TITLE_TAG,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    cand_tokens = tokenizer.tokenize(candidate_desc)
    if candidate_title is not None:
        title_tokens = tokenizer.tokenize(candidate_title)
        cand_tokens = title_tokens + [title_tag] + cand_tokens

    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [cls_token] + cand_tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }


def make_dict_from_jsonl(jsonl_file, key_field, value_field=None):
    out_dict = {}
    with io.open(jsonl_file, mode="r", encoding="utf-8") as file:
        for line in file:
            line = json.loads(line)
            if value_field is not None:
                out_dict[line[key_field]] = line[value_field]
            else:
                out_dict[line[key_field]] = line
    return out_dict


def process_mention_data(
        samples,
        tokenizer,
        max_context_length,
        max_cand_length,
        silent,
        mention_key="mention",
        context_key="context",
        label_key="label",
        title_key="label_title",
        ent_start_token=ENT_START_TAG,
        ent_end_token=ENT_END_TAG,
        title_token=ENT_TITLE_TAG,
        debug=False,
        logger=None,
        hard_negatives_file=None,
        entities_file=None,
        max_num_negatives=9,
):
    num_examples_skipped = 0
    num_times_entity_not_found = 0

    entity_dict = None
    id_to_negative = None

    processed_candidate_cache = {}

    if hard_negatives_file is not None:
        assert entities_file is not None

        logger.info("Reading the entities file")
        entity_dict = make_dict_from_jsonl(
            jsonl_file=entities_file, key_field="title", value_field="text"
        )

        id_to_negative = make_dict_from_jsonl(
            jsonl_file=hard_negatives_file,
            key_field="id",
            value_field="top_10_predictions",
        )

        processed_candidate_cache = {}

    processed_samples = []

    if debug:
        samples = samples[:200]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    use_world = True

    for idx, sample in enumerate(iter_):

        # skip training example if hard negative is not available
        if id_to_negative is not None:
            if sample["id"] not in id_to_negative:
                num_examples_skipped += 1
                continue

        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
        )

        label = sample[label_key]
        title = sample.get(title_key, None)

        if id_to_negative is not None:
            sample_id = sample["id"]
            label_tokens = []

            # put the positive in the first place.
            if title in processed_candidate_cache:
                lbl_tok = processed_candidate_cache[title]
            else:
                lbl_tok = get_candidate_representation(
                    label, tokenizer, max_cand_length, title,
                )["ids"]
                processed_candidate_cache[title] = lbl_tok

            label_tokens.append(lbl_tok)

            for neg_title in id_to_negative[sample_id]:
                if len(label_tokens) == max_num_negatives + 1:
                    break
                if neg_title == title:
                    continue

                try:
                    ent_desc = entity_dict[neg_title]
                except KeyError:
                    ent_desc = ""
                    num_times_entity_not_found += 1

                if neg_title in processed_candidate_cache:
                    lbl_tok = processed_candidate_cache[neg_title]
                else:
                    lbl_tok = get_candidate_representation(
                        ent_desc, tokenizer, max_cand_length, neg_title,
                    )["ids"]
                    processed_candidate_cache[neg_title] = lbl_tok
                label_tokens.append(lbl_tok)

            assert len(label_tokens) == max_num_negatives + 1

        else:
            label_tokens = get_candidate_representation(
                label, tokenizer, max_cand_length, title,
            )["ids"]

        label_idx = int(sample["label_id"])

        record = {
            "context": context_tokens,
            "label": label_tokens,
            "label_idx": [label_idx],
        }

        if "world" in sample:
            src = sample["world"]
            src = world_to_id[src]
            record["src"] = [src]
            use_world = True
        else:
            use_world = False

        processed_samples.append(record)

    # if debug and logger:
    #     logger.info("====Processed samples: ====")
    #     for sample in processed_samples[:5]:
    #         logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
    #         logger.info(
    #             "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
    #         )
    #         logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
    #         logger.info(
    #             "Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]])
    #         )
    #         #             logger.info("Src : %d" % sample["src"][0])
    #         logger.info("Label_id : %d" % sample["label_idx"][0])

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_samples, "label"), dtype=torch.long,
    )
    if use_world:
        src_vecs = torch.tensor(
            select_field(processed_samples, "src"), dtype=torch.long,
        )
    label_idx = torch.tensor(
        select_field(processed_samples, "label_idx"), dtype=torch.long,
    )
    data = {
        "context_vecs": context_vecs,
        "cand_vecs": cand_vecs,
        "label_idx": label_idx,
    }

    if use_world:
        data["src"] = src_vecs
        tensor_data = TensorDataset(context_vecs, cand_vecs, src_vecs, label_idx)
    else:
        tensor_data = TensorDataset(context_vecs, cand_vecs, label_idx)

    if id_to_negative is not None:
        logger.info(
            "Skipped {} out of {} samples".format(num_examples_skipped, len(samples))
        )
        logger.info(
            "Entity description not found {} times".format(num_times_entity_not_found)
        )

    return data, tensor_data


def get_negative_samples(positive_samples, num_types, num_neg_samples):
    negative_samples = []

    if num_neg_samples == 0:
        return negative_samples

    samples_collected_so_far = set(positive_samples)
    while len(negative_samples) < num_neg_samples:
        random_integer = np.random.randint(0, num_types)
        if random_integer not in samples_collected_so_far:
            samples_collected_so_far.add(random_integer)
            negative_samples.append(random_integer)

    assert len(negative_samples) == num_neg_samples

    return negative_samples


def prepare_type_list_and_labels(sample, types_key, max_type_list_len, num_types):
    positive_types = sample[types_key]

    if type(positive_types) == list:
        positive_types = positive_types[:max_type_list_len]
    else:
        # sometimes we have sample[types_key] = {} instead of []
        positive_types = []

    num_positive_types = len(positive_types)
    num_negative_types = max_type_list_len - num_positive_types

    negative_samples = get_negative_samples(
        positive_samples=positive_types,
        num_types=num_types,
        num_neg_samples=num_negative_types,
    )

    out_type_list = positive_types + negative_samples
    out_type_labels = [1.0] * num_positive_types + [0.0] * num_negative_types

    assert len(out_type_list) == max_type_list_len == len(out_type_labels)

    return out_type_list, out_type_labels


def process_mention_data_with_types(
        samples,
        tokenizer,
        max_context_length,
        max_cand_length,
        silent,
        max_type_list_len,
        num_types,
        types_key="fgetc_category_id",
        mention_key="mention",
        context_key="context",
        label_key="label",
        title_key="label_title",
        ent_start_token=ENT_START_TAG,
        ent_end_token=ENT_END_TAG,
        title_token=ENT_TITLE_TAG,
        debug=False,
        logger=None,
        hard_negatives_file=None,
        entities_file=None,
        max_num_negatives=9,
):
    num_examples_skipped = 0
    num_times_entity_not_found = 0

    entity_dict = None
    id_to_negative = None

    processed_candidate_cache = {}

    if hard_negatives_file is not None:
        assert entities_file is not None

        logger.info("Reading the entities file")
        entity_dict = make_dict_from_jsonl(
            jsonl_file=entities_file, key_field="title", value_field="text"
        )

        id_to_negative = make_dict_from_jsonl(
            jsonl_file=hard_negatives_file,
            key_field="id",
            value_field="top_10_predictions",
        )

        processed_candidate_cache = {}

    processed_samples = []

    if debug:
        samples = samples[:200]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    use_world = True

    for idx, sample in enumerate(iter_):

        # skip training example if hard negative is not available
        if id_to_negative is not None:
            if sample["id"] not in id_to_negative:
                num_examples_skipped += 1
                continue

        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
        )

        label = sample[label_key]
        title = sample.get(title_key, None)

        if id_to_negative is not None:
            sample_id = sample["id"]
            label_tokens = []

            # put the positive in the first place.
            if title in processed_candidate_cache:
                lbl_tok = processed_candidate_cache[title]
            else:
                lbl_tok = get_candidate_representation(
                    label, tokenizer, max_cand_length, title,
                )["ids"]
                processed_candidate_cache[title] = lbl_tok

            label_tokens.append(lbl_tok)

            for neg_title in id_to_negative[sample_id]:
                if len(label_tokens) == max_num_negatives + 1:
                    break
                if neg_title == title:
                    continue

                try:
                    ent_desc = entity_dict[neg_title]
                except KeyError:
                    ent_desc = ""
                    num_times_entity_not_found += 1

                if neg_title in processed_candidate_cache:
                    lbl_tok = processed_candidate_cache[neg_title]
                else:
                    lbl_tok = get_candidate_representation(
                        ent_desc, tokenizer, max_cand_length, neg_title,
                    )["ids"]
                    processed_candidate_cache[neg_title] = lbl_tok
                label_tokens.append(lbl_tok)

            assert len(label_tokens) == max_num_negatives + 1

        else:
            label_tokens = get_candidate_representation(
                label, tokenizer, max_cand_length, title,
            )["ids"]

        label_idx = int(sample["label_id"])

        type_list, type_labels = prepare_type_list_and_labels(
            sample, types_key, max_type_list_len, num_types
        )

        record = {
            "context": context_tokens,
            "label": label_tokens,
            "label_idx": [label_idx],
            "type_list": type_list,
            "type_labels": type_labels,
        }

        if "world" in sample:
            src = sample["world"]
            src = world_to_id[src]
            record["src"] = [src]
            use_world = True
        else:
            use_world = False

        processed_samples.append(record)

    # if debug and logger:
    #     logger.info("====Processed samples: ====")
    #     for sample in processed_samples[:5]:
    #         logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
    #         logger.info(
    #             "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
    #         )
    #         logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
    #         logger.info(
    #             "Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]])
    #         )
    #         #             logger.info("Src : %d" % sample["src"][0])
    #         logger.info("Label_id : %d" % sample["label_idx"][0])

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_samples, "label"), dtype=torch.long,
    )
    if use_world:
        src_vecs = torch.tensor(
            select_field(processed_samples, "src"), dtype=torch.long,
        )
    label_idx = torch.tensor(
        select_field(processed_samples, "label_idx"), dtype=torch.long,
    )

    type_vec = torch.tensor(
        select_field(processed_samples, "type_list"), dtype=torch.long,
    )
    type_label_vec = torch.tensor(select_field(processed_samples, "type_labels"))

    data = {
        "context_vecs": context_vecs,
        "cand_vecs": cand_vecs,
        "label_idx": label_idx,
        "type_vec": type_vec,
        "type_label_vec": type_label_vec,
    }

    tensor_data = TensorDataset(
        context_vecs, cand_vecs, label_idx, type_vec, type_label_vec
    )

    if id_to_negative is not None:
        logger.info(
            "Skipped {} out of {} samples".format(num_examples_skipped, len(samples))
        )
        logger.info(
            "Entity description not found {} times".format(num_times_entity_not_found)
        )

    return data, tensor_data


def process_mention_data_with_types_inference(
        samples,
        tokenizer,
        max_context_length,
        max_cand_length,
        silent,
        max_type_list_len,
        num_types,
        types_key="fgetc_category_id",
        mention_key="mention",
        context_key="context",
        label_key="label",
        title_key="label_title",
        ent_start_token=ENT_START_TAG,
        ent_end_token=ENT_END_TAG,
        title_token=ENT_TITLE_TAG,
        debug=False,
        logger=None,
        hard_negatives_file=None,
        entities_file=None,
        max_num_negatives=9,
        positive_types_strategy="lflc_ancestor",
):
    num_examples_skipped = 0
    num_times_entity_not_found = 0

    entity_dict = None
    id_to_negative = None

    processed_candidate_cache = {}

    processed_samples = []

    if debug:
        samples = samples[:200]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    use_world = True

    for idx, sample in enumerate(iter_):

        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
        )

        label = sample[label_key]
        title = sample.get(title_key, None)

        label_tokens = get_candidate_representation(
            label, tokenizer, max_cand_length, title,
        )["ids"]

        label_idx = int(sample["label_id"])

        type_labels = prepare_type_list_and_labels_inference(
            sample, types_key, num_types, num_types, positive_types_strategy
        )

        if type(sample["id"]) == str:
            sample["id"] = int(sample["id"].replace("_", ""))

        record = {
            "context": context_tokens,
            "label": label_tokens,
            "label_idx": [label_idx],
            "type_labels": type_labels,
            "id": [sample["id"]],
        }

        if "world" in sample:
            src = sample["world"]
            src = world_to_id[src]
            record["src"] = [src]
            use_world = True
        else:
            use_world = False

        processed_samples.append(record)

    # if debug and logger:
    #     logger.info("====Processed samples: ====")
    #     for sample in processed_samples[:5]:
    #         logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
    #         logger.info(
    #             "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
    #         )
    #         logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
    #         logger.info(
    #             "Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]])
    #         )
    #         #             logger.info("Src : %d" % sample["src"][0])
    #         logger.info("Label_id : %d" % sample["label_idx"][0])

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_samples, "label"), dtype=torch.long,
    )
    if use_world:
        src_vecs = torch.tensor(
            select_field(processed_samples, "src"), dtype=torch.long,
        )
    label_idx = torch.tensor(
        select_field(processed_samples, "label_idx"), dtype=torch.long,
    )
    ids_topk = torch.tensor(select_field(processed_samples, "id"), dtype=torch.long, )

    type_label_vec = torch.tensor(select_field(processed_samples, "type_labels"))

    data = {
        "context_vecs": context_vecs,
        "cand_vecs": cand_vecs,
        "label_idx": label_idx,
        "type_label_vec": type_label_vec,
        "ids": ids_topk,
    }

    tensor_data = TensorDataset(
        context_vecs, cand_vecs, label_idx, type_label_vec, ids_topk
    )

    return data, tensor_data


def prepare_type_list_and_labels_inference(
        sample,
        types_key,
        max_type_list_len,
        num_types,
        positive_types_strategy="lflc_ancestor",
):
    positive_types = sample[types_key]

    if type(positive_types) != list:
        # sometimes we have sample[types_key] = {} instead of []
        positive_types = []

    # take the union of DBPedia fine and coarse types
    if types_key == "fine_types_id":
        positive_types += sample["coarse_types_id"]
        if positive_types_strategy == "lflc_ancestor":
            positive_types += sample["ancestors_types_id"]

    out_type_labels = [0] * num_types
    for index in positive_types:
        out_type_labels[index] = 1
    return out_type_labels


def process_mention_data_with_types_2(
        samples,
        tokenizer,
        max_context_length,
        max_cand_length,
        silent,
        max_type_list_len=-1,
        num_types=-1,
        types_key="fgetc_category_id",
        mention_key="mention",
        context_key="context",
        label_key="label",
        title_key="label_title",
        ent_start_token=ENT_START_TAG,
        ent_end_token=ENT_END_TAG,
        title_token=ENT_TITLE_TAG,
        debug=False,
        logger=None,
        hard_negatives_file=None,
        entities_file=None,
        max_num_negatives=9,
        positive_types_strategy="lflc_ancestor",
):
    num_examples_skipped = 0
    num_times_entity_not_found = 0

    entity_dict = None
    id_to_negative = None

    processed_candidate_cache = {}

    if hard_negatives_file is not None:
        assert entities_file is not None

        logger.info("Reading the entities file")
        entity_dict = make_dict_from_jsonl(
            jsonl_file=entities_file, key_field="title", value_field="text"
        )

        id_to_negative = make_dict_from_jsonl(
            jsonl_file=hard_negatives_file,
            key_field="id",
            value_field="top_10_predictions",
        )

        processed_candidate_cache = {}

    processed_samples = []

    if debug:
        samples = samples[:200]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    use_world = True

    for idx, sample in enumerate(iter_):

        # skip training example if hard negative is not available
        if id_to_negative is not None:
            if sample["id"] not in id_to_negative:
                num_examples_skipped += 1
                continue

        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
        )

        label = sample[label_key]
        title = sample.get(title_key, None)

        if id_to_negative is not None:
            sample_id = sample["id"]
            label_tokens = []

            # put the positive in the first place.
            if title in processed_candidate_cache:
                lbl_tok = processed_candidate_cache[title]
            else:
                lbl_tok = get_candidate_representation(
                    label, tokenizer, max_cand_length, title,
                )["ids"]
                processed_candidate_cache[title] = lbl_tok

            label_tokens.append(lbl_tok)

            for neg_title in id_to_negative[sample_id]:
                if len(label_tokens) == max_num_negatives + 1:
                    break
                if neg_title == title:
                    continue

                try:
                    ent_desc = entity_dict[neg_title]
                except KeyError:
                    ent_desc = ""
                    num_times_entity_not_found += 1

                if neg_title in processed_candidate_cache:
                    lbl_tok = processed_candidate_cache[neg_title]
                else:
                    lbl_tok = get_candidate_representation(
                        ent_desc, tokenizer, max_cand_length, neg_title,
                    )["ids"]
                    processed_candidate_cache[neg_title] = lbl_tok
                label_tokens.append(lbl_tok)

            assert len(label_tokens) == max_num_negatives + 1

        else:
            label_tokens = get_candidate_representation(
                label, tokenizer, max_cand_length, title,
            )["ids"]

        label_idx = int(sample["label_id"])

        type_labels = prepare_type_list_and_labels_inference(
            sample, types_key, max_type_list_len, num_types, positive_types_strategy
        )

        if type(sample["id"]) == str:
            sample["id"] = int(sample["id"].replace("_", ""))

        record = {
            "context": context_tokens,
            "label": label_tokens,
            "label_idx": [label_idx],
            "type_labels": type_labels,
            "id": [sample["id"]],
        }

        if "world" in sample:
            src = sample["world"]
            src = world_to_id[src]
            record["src"] = [src]
            use_world = True
        else:
            use_world = False

        processed_samples.append(record)

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_samples, "label"), dtype=torch.long,
    )
    if use_world:
        src_vecs = torch.tensor(
            select_field(processed_samples, "src"), dtype=torch.long,
        )
    label_idx = torch.tensor(
        select_field(processed_samples, "label_idx"), dtype=torch.long,
    )

    ids_topk = torch.tensor(select_field(processed_samples, "id"), dtype=torch.long)

    type_label_vec = torch.tensor(
        select_field(processed_samples, "type_labels"), dtype=torch.float32
    )

    data = {
        "context_vecs": context_vecs,
        "cand_vecs": cand_vecs,
        "label_idx": label_idx,
        "type_label_vec": type_label_vec,
        "ids": ids_topk,
    }

    tensor_data = TensorDataset(
        context_vecs, cand_vecs, label_idx, type_label_vec, ids_topk
    )

    if id_to_negative is not None:
        logger.info(
            "Skipped {} out of {} samples".format(num_examples_skipped, len(samples))
        )
        logger.info(
            "Entity description not found {} times".format(num_times_entity_not_found)
        )

    return data, tensor_data


def prepare_type_list_and_labels_inference_2(
        sample,
        types_key,
        num_types,
        positive_types_strategy="lflc_ancestor",
):
    out_type_labels = prepare_type_list_and_labels_inference(
        sample,
        types_key,
        None,
        num_types,
        positive_types_strategy=positive_types_strategy,
    )

    descendants_mask = [0] * num_types
    for index in sample["descendants_types_id"]:
        descendants_mask[index] = 1

    return [out_type_labels, descendants_mask]


def process_mention_data_with_types_3(
        samples,
        tokenizer,
        max_context_length,
        max_cand_length,
        silent,
        max_type_list_len=-1,
        num_types=-1,
        types_key="fgetc_category_id",
        mention_key="mention",
        context_key="context",
        label_key="label",
        title_key="label_title",
        ent_start_token=ENT_START_TAG,
        ent_end_token=ENT_END_TAG,
        title_token=ENT_TITLE_TAG,
        debug=False,
        logger=None,
        hard_negatives_file=None,
        entities_file=None,
        max_num_negatives=9,
        positive_types_strategy="lflc_ancestor",
):
    num_examples_skipped = 0
    num_times_entity_not_found = 0

    entity_dict = None
    id_to_negative = None

    processed_candidate_cache = {}

    if hard_negatives_file is not None:
        assert entities_file is not None

        logger.info("Reading the entities file")
        entity_dict = make_dict_from_jsonl(
            jsonl_file=entities_file, key_field="title", value_field="text"
        )

        id_to_negative = make_dict_from_jsonl(
            jsonl_file=hard_negatives_file,
            key_field="id",
            value_field="top_10_predictions",
        )

        processed_candidate_cache = {}

    processed_samples = []

    if debug:
        samples = samples[:200]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    use_world = True

    for idx, sample in enumerate(iter_):

        # skip training example if hard negative is not available
        if id_to_negative is not None:
            if sample["id"] not in id_to_negative:
                num_examples_skipped += 1
                continue

        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
        )

        label = sample[label_key]
        title = sample.get(title_key, None)

        if id_to_negative is not None:
            sample_id = sample["id"]
            label_tokens = []

            # put the positive in the first place.
            if title in processed_candidate_cache:
                lbl_tok = processed_candidate_cache[title]
            else:
                lbl_tok = get_candidate_representation(
                    label, tokenizer, max_cand_length, title,
                )["ids"]
                processed_candidate_cache[title] = lbl_tok

            label_tokens.append(lbl_tok)

            for neg_title in id_to_negative[sample_id]:
                if len(label_tokens) == max_num_negatives + 1:
                    break
                if neg_title == title:
                    continue

                try:
                    ent_desc = entity_dict[neg_title]
                except KeyError:
                    ent_desc = ""
                    num_times_entity_not_found += 1

                if neg_title in processed_candidate_cache:
                    lbl_tok = processed_candidate_cache[neg_title]
                else:
                    lbl_tok = get_candidate_representation(
                        ent_desc, tokenizer, max_cand_length, neg_title,
                    )["ids"]
                    processed_candidate_cache[neg_title] = lbl_tok
                label_tokens.append(lbl_tok)

            assert len(label_tokens) == max_num_negatives + 1

        else:
            label_tokens = get_candidate_representation(
                label, tokenizer, max_cand_length, title,
            )["ids"]

        label_idx = int(sample["label_id"])

        type_labels = prepare_type_list_and_labels_inference_2(
            sample, types_key, num_types, positive_types_strategy
        )

        if type(sample["id"]) == str:
            sample["id"] = int(sample["id"].replace("_", ""))

        record = {
            "context": context_tokens,
            "label": label_tokens,
            "label_idx": [label_idx],
            "type_labels": type_labels,
            "id": [sample["id"]],
        }

        if "world" in sample:
            src = sample["world"]
            src = world_to_id[src]
            record["src"] = [src]
            use_world = True
        else:
            use_world = False

        processed_samples.append(record)

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_samples, "label"), dtype=torch.long,
    )
    if use_world:
        src_vecs = torch.tensor(
            select_field(processed_samples, "src"), dtype=torch.long,
        )
    label_idx = torch.tensor(
        select_field(processed_samples, "label_idx"), dtype=torch.long,
    )

    ids_topk = torch.tensor(select_field(processed_samples, "id"), dtype=torch.long)

    type_label_vec = torch.tensor(
        select_field(processed_samples, "type_labels"), dtype=torch.float32
    )

    data = {
        "context_vecs": context_vecs,
        "cand_vecs": cand_vecs,
        "label_idx": label_idx,
        "type_label_vec": type_label_vec,
        "ids": ids_topk,
    }

    tensor_data = TensorDataset(
        context_vecs, cand_vecs, label_idx, type_label_vec, ids_topk
    )

    if id_to_negative is not None:
        logger.info(
            "Skipped {} out of {} samples".format(num_examples_skipped, len(samples))
        )
        logger.info(
            "Entity description not found {} times".format(num_times_entity_not_found)
        )

    return data, tensor_data


def filter_empty_types(data_in):
    data_out = []
    for data_item in data_in:
        type_set = data_item["fine_types_id"] + data_item["coarse_types"] + data_item["ancestors_types_id"]
        if any(type_set):
            data_out.append(data_item)
    return data_out


def process_typed_data_factory(model_number, params, samples, tokenizer, logger):
    if model_number == 1:
        data, tensor_data = process_mention_data_with_types(
            samples=samples,
            tokenizer=tokenizer,
            max_context_length=params["max_context_length"],
            max_cand_length=params["max_cand_length"],
            silent=params["silent"],
            max_type_list_len=params["max_type_list_len"],
            num_types=params["num_types"],
            types_key=params["types_key"],
            context_key=params["context_key"],
            label_key="label",
            debug=params["debug"],
            logger=logger,
            hard_negatives_file=params["hard_negatives_file"],
            entities_file=params["entities_file"],
            max_num_negatives=params["max_num_negatives"],
        )
    elif model_number in [2, 4]:
        data, tensor_data = process_mention_data_with_types_2(
            samples=samples,
            tokenizer=tokenizer,
            max_context_length=params["max_context_length"],
            max_cand_length=params["max_cand_length"],
            silent=params["silent"],
            max_type_list_len=params["max_type_list_len"],
            num_types=params["num_types"],
            types_key=params["types_key"],
            context_key=params["context_key"],
            label_key="label",
            debug=params["debug"],
            logger=logger,
            hard_negatives_file=params["hard_negatives_file"],
            entities_file=params["entities_file"],
            max_num_negatives=params["max_num_negatives"],
            positive_types_strategy=params["positive_types"],
        )
    elif model_number in [3, 5]:
        data, tensor_data = process_mention_data_with_types_3(
            samples=samples,
            tokenizer=tokenizer,
            max_context_length=params["max_context_length"],
            max_cand_length=params["max_cand_length"],
            silent=params["silent"],
            max_type_list_len=params["max_type_list_len"],
            num_types=params["num_types"],
            types_key=params["types_key"],
            context_key=params["context_key"],
            label_key="label",
            debug=params["debug"],
            logger=logger,
            hard_negatives_file=params["hard_negatives_file"],
            entities_file=params["entities_file"],
            max_num_negatives=params["max_num_negatives"],
            positive_types_strategy=params["positive_types"],
        )
    else:
        assert False, "Unsupported value passed for model_number"

    return data, tensor_data
