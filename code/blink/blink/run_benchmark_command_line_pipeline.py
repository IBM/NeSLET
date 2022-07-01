import os
import json
import argparse
import prettytable

import blink.main_dense as main_dense
import blink.candidate_ranking.utils as utils


parser = argparse.ArgumentParser()
parser.add_argument("--biencoder_model", type=str, help="Path to the biencoder model.")
parser.add_argument(
    "--biencoder_config", type=str, help="Path to the biencoder configuration."
)
parser.add_argument("--biencoder_training_params", type=str)
parser.add_argument("--entity_encoding", type=str, help="Path to the entity catalogue.")
#################################################
parser.add_argument(
    "--entity_catalogue",
    type=str,
    default="NeSLET_everything/scratch-shared/facebook_original_models/entity.jsonl",
    help="Path to the entity catalogue.",
)
# crossencoder
parser.add_argument(
    "--crossencoder_model",
    type=str,
    default="NeSLET_everything/scratch-shared/facebook_original_models/crossencoder_wiki_large.bin",
    help="Path to the crossencoder model.",
)
parser.add_argument(
    "--crossencoder_config",
    type=str,
    default="NeSLET_everything/scratch-shared/facebook_original_models/crossencoder_wiki_large.json",
    help="Path to the crossencoder configuration.",
)
parser.add_argument(
    "--top_k",
    type=int,
    default=100,
    help="Number of candidates retrieved by biencoder.",
)
parser.add_argument(
    "--faiss_index", type=str, default=None, help="whether to use faiss index"
)
parser.add_argument("--index_path", type=str, default=None, help="path to load indexer")
parser.add_argument(
    "--fast",
    dest="fast",
    default=False,
    action="store_true",
    help="only biencoder mode",
)
parser.add_argument("--test_mentions", default=None)
parser.add_argument("--test_entities", default=None)
parser.add_argument(
    "--interactive", "-i", default=False, action="store_true", help="Interactive mode."
)
#################################################
parser.add_argument("--output_path", type=str, help="Path to the output.")
parser.add_argument("--dataset_name", type=str, help="Dataset Name")
parser.add_argument("--test_file_path", type=str, help="Test file path")
parser.add_argument("--output_file_path", type=str, help="Output prediction file path")
parser.add_argument(
    "--output_score_file_path", type=str, help="Output prediction file path"
)


# parser.add_argument("--prediction_file")

args_1 = parser.parse_args()
# PARAMETERS= parser.parse_args().__dict__
# args = argparse.Namespace(**PARAMETERS)

DATASETS = [{"name": args_1.dataset_name, "filename": args_1.test_file_path}]


logger = utils.get_logger(args_1.output_path)

models = main_dense.load_models(args_1, logger)

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
    PARAMETERS = args_1.__dict__
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
        test_prediction_score,
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

with open(args_1.output_file_path, "w") as write_file:
    json.dump(predictions, write_file)


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = "a+" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + "\n")
    print("Wrote {} records to {}".format(len(data), output_path))


dump_jsonl(test_prediction_score, args_1.output_score_file_path)

