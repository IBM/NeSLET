# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import argparse
import prettytable

import blink.main_dense as main_dense
import blink.candidate_ranking.utils as utils

DATASETS = [{"name": "unseen mention-from fgET","filename":"data/BLINK_benchmark/unseen_mention_test_100_examples.jsonl"}]

#    {
#        "name": "AIDA-YAGO2 testa",
#        "filename": "data/BLINK_benchmark/AIDA-YAGO2_testa.jsonl",
#    },
#    {
#        "name": "AIDA-YAGO2 testb",
#        "filename": "data/BLINK_benchmark/AIDA-YAGO2_testb.jsonl",
#    },
#    {"name": "ACE 2004", "filename": "data/BLINK_benchmark/ace2004_questions.jsonl"},
#    {"name": "aquaint", "filename": "data/BLINK_benchmark/aquaint_questions.jsonl"},
#    {
#        "name": "clueweb - WNED-CWEB (CWEB)",
#        "filename": "data/BLINK_benchmark/clueweb_questions.jsonl",
#    },
#    {"name": "msnbc", "filename": "data/BLINK_benchmark/msnbc_questions.jsonl"},
#    {
#        "name": "wikipedia - WNED-WIKI (WIKI)",
#        "filename": "data/BLINK_benchmark/wnedwiki_questions.jsonl",
#    },
#     {
#         "name": "unseen mention-from fgET",
#         "filename":"data/BLINK_benchmark/unseen_mention_test.jsonl"
#     } 
#]

PARAMETERS = {
    "faiss_index": None,
    #"no_cuda": False,
    "index_path": None,
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "biencoder_model": "models/biencoder_wiki_large.bin",
    #"biencoder_model":"NeSLET_everything/NeSLET/code/blink/blink/trained_models/25percent/epoch_0/pytorch_model.bin",
    "biencoder_config": "models/biencoder_wiki_large.json",
    # "biencoder_config":"NeSLET_everything/NeSLET/code/blink/blink/trained_models/25percent/epoch_0/config.json",
    #"biencoder_config":"NeSLET_everything/NeSLET/code/blink/blink/trained_models/training_params_old.json",
    #"entity_catalogue":"models/entities_512.jsonl",
    "entity_catalogue": "models/entity.jsonl",
    "entity_encoding": "models/all_entities_large.t7",
    #"entity_encoding":"NeSLET_everything/NeSLET/code/blink/blink/trained_models/25percent/epoch_0/tiny.t7",
    "crossencoder_model": "models/crossencoder_wiki_large.bin",
    "crossencoder_config": "models/crossencoder_wiki_large.json",
    "output_path": "output",
    "fast": False,
    "top_k": 10,
}
args = argparse.Namespace(**PARAMETERS)

logger = utils.get_logger(args.output_path)

models = main_dense.load_models(args, logger)

table = prettytable.PrettyTable(
    [
        "DATASET",
        "biencoder accuracy",
        "recall at 100",
        "crossencoder normalized accuracy",
        "overall unormalized accuracy",
        "support",
    ]
)

for dataset in DATASETS:
    logger.info(dataset["name"])
    PARAMETERS["test_mentions"] = dataset["filename"]

    args = argparse.Namespace(**PARAMETERS)
    (
        biencoder_accuracy,
        recall_at,
        crossencoder_normalized_accuracy,
        overall_unormalized_accuracy,
        num_datapoints,
        predictions,
        scores,
    ) = main_dense.run(args, logger, *models)


    table.add_row(
        [
            dataset["name"],
            round(biencoder_accuracy, 4),
            round(recall_at, 4),
            round(crossencoder_normalized_accuracy, 4),
            round(overall_unormalized_accuracy, 4),
            num_datapoints,
        ]
    )

logger.info("\n{}".format(table))

with open("data/unseen_prediction_file.json", "w") as write_file:
    json.dump(predictions, write_file)
