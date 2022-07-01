import json
import glob
import argparse
import os
import prettytable


class ResultGatherer:
    def __init__(self, config):
        self.config = config

        self.fname_to_set_map = {
            "unseen_60K_mention_dev_10K_dbpedia_ancs_desc.jsonl": "um_val",
            "unseen_60K_mention_test_10K_updated_dbpedia_ancs_desc.jsonl": "um_test",
            "conll_60K_mention_dev_4791_dbpedia_types_desc_ances.jsonl": "conll_val",
            "conll_60K_mention_test_4485_updated_dbpedia_types_desc_ances.jsonl": "conll_test"
        }

        # self.f1_key = "corrected_avg_f1"
        self.f1_key = "avg_f1"

        self.metrics_file_name = "eval_set_metrics.json"

    def read_json(self, file_path):
        with open(file_path) as in_file:
            data = json.load(in_file)
        return data

    def get_files_to_read(self):
        # path_template = os.path.join(self.config["base_dir"], "*", self.metrics_file_name)
        path_template = os.path.join(self.config["base_dir"], self.config["exp_name_template"], self.metrics_file_name)
        files_to_read = glob.glob(path_template)
        return files_to_read

    def get_metrics_from_file(self, file_path):
        # Data format:
        # all_metrics = {"new_um_valid_dbpedia_type_desc_ance.jsonl": {"corrected_avg_f1": 0.25982342969425243, "avg_f1": 0.3506184223184241, "normalized_accuracy": 0.25982342969425243}, "unseen_60K_mention_dev_10K_dbpedia_ancs_desc.jsonl": {"corrected_avg_f1": 0.19528940021898164, "avg_f1": 0.33152690476190805, "normalized_accuracy": 0.19528940021898164}, "unseen_60K_mention_test_10K_updated_dbpedia_ancs_desc.jsonl": {"corrected_avg_f1": 0.20628790889768597, "avg_f1": 0.34304987413467863, "normalized_accuracy": 0.20628790889768597}}

        all_metrics = self.read_json(file_path)

        val_and_test_metrics = {}

        for file_name, metrics in all_metrics.items():
            if file_name in self.fname_to_set_map.keys():
                set_name = self.fname_to_set_map[file_name]
                f1 = metrics[self.f1_key]
                val_and_test_metrics[set_name] = f1

        return val_and_test_metrics

    def print_results(self):
        files_to_read = self.get_files_to_read()

        table = prettytable.PrettyTable(
            [
                "Experiment",
                "um_val",
                "um_test",
                "conll_val",
                "conll_test"
            ]
        )

        all_metrics = []
        for file_to_read in files_to_read:
            exp_name = os.path.basename(os.path.dirname(file_to_read))
            metrics = self.get_metrics_from_file(file_path=file_to_read)

            table.add_row(
                [
                    exp_name,
                    metrics.get("um_val", "-1"),
                    metrics.get("um_test", "-1"),
                    metrics.get("conll_val", "-1"),
                    metrics.get("conll_test", "-1"),
                ]
            )

        print(table)



if __name__ == "__main__":
    """
    python gather_type_model_results.py --base_dir /type_models/models
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str)
    parser.add_argument("--exp_name_template", type=str, default="*")

    command_line_args = parser.parse_args()
    command_line_args = command_line_args.__dict__

    result_gatherer = ResultGatherer(command_line_args)
    result_gatherer.print_results()
