import json
import argparse
import prettytable

import blink.main_dense as main_dense
import blink.candidate_ranking.utils as utils

DATASETS = [
    {
        "name": "wikipedia - WNED-WIKI (WIKI) 10K ",
        #"name": "Conll-Yago-4485-test",
        #"filename": "zel_everything/scratch-shared/data/fget_processed_data/conll_fget_data_processed/conll_60K_mention_test_4485.json"
        "filename": "zel_everything/scratch-shared/data/fget_processed_data/unseen_mention_fget_data_processed/unseen_60K_mention_test_10K.jsonl"
    }]
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
#    }

parser = argparse.ArgumentParser()
parser.add_argument("--biencoder_model",type=str, help="Path to the biencoder model." )
parser.add_argument("--biencoder_config", type=str,help="Path to the biencoder configuration.")
parser.add_argument("--biencoder_training_params", type=str)
parser.add_argument("--entity_encoding",type=str,help="Path to the entity catalogue.")
#################################################
parser.add_argument("--entity_catalogue",type=str,default="zel_everything/scratch-shared/facebook_original_models/entity.jsonl", help="Path to the entity catalogue.")
 # crossencoder
parser.add_argument("--crossencoder_model",type=str, default="zel_everything/scratch-shared/facebook_original_models/crossencoder_wiki_large.bin", help="Path to the crossencoder model.")
parser.add_argument("--crossencoder_config",type=str,default="zel_everything/scratch-shared/facebook_original_models/crossencoder_wiki_large.json",help="Path to the crossencoder configuration.")
parser.add_argument("--top_k", type=int, default=100, help="Number of candidates retrieved by biencoder.")
parser.add_argument("--faiss_index", type=str, default=None, help="whether to use faiss index")
parser.add_argument("--index_path", type=str, default=None, help="path to load indexer")
parser.add_argument("--fast", dest="fast", action="store_true", help="only biencoder mode")
parser.add_argument("--test_mentions",default=None)
parser.add_argument("--test_entities",default=None)
parser.add_argument("--interactive","-i", default= False, action="store_true", help="Interactive mode.")
#################################################
parser.add_argument("--output_path",type=str,help="Path to the output.")

args_1 = parser.parse_args()

#args = argparse.Namespace(**PARAMETERS)

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


