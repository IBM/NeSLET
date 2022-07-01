import argparse
import json

import torch
from transformers import pipeline

import re


def read_json(path):
    with open(path) as in_file:
        data = json.load(in_file)
    return data


class BertFeatureExtractor:
    def __init__(self):
        self.feature_extractor = pipeline(
            "feature-extraction", model="bert-base-uncased", framework="pt"
        )

    def camel_case_split(self, string_in):
        matches = re.finditer(
            ".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", string_in
        )
        return [m.group(0) for m in matches]

    def normalize_string_dbpedia(self, str_in):
        return " ".join(self.camel_case_split(str_in[str_in.rfind("/") + 1 :]))

    def normalize_string(self, str_in):
        if "dbpedia" in str_in:
            return self.normalize_string_dbpedia(str_in)
        else:
            return str_in

    def extract_features(self, inputs):
        out_features = []

        inputs = [self.normalize_string(s) for s in inputs]

        # results is a list of list of list. Dimensions: (Samples, Tokens, Vector Size)
        results = self.feature_extractor(inputs)

        assert len(results) == len(inputs)

        for res in results:
            # ignore the [CLS], average the rest
            word_vectors = torch.tensor(res[1:])
            f = torch.mean(word_vectors, dim=0)
            out_features.append(f.unsqueeze(0))

        out_features = torch.cat(out_features, dim=0)

        assert out_features.shape[0] == len(inputs)
        assert out_features.shape[1] == 768

        return out_features


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str)
    parser.add_argument("--output_file_path", type=str)

    config = parser.parse_args()

    data_in = read_json(config.input_file_path)

    types_list = []

    if type(list(data_in.keys())[0]) is not int:
        data_in = {int(key): value for key, value in data_in.items()}

    for i in range(len(data_in)):
        types_list.append(data_in[i])

    feature_extractor = BertFeatureExtractor()

    features = feature_extractor.extract_features(types_list)
    torch.save(features, config.output_file_path)

    print("Feature matrix size: {}".format(features.shape))
    print("Done")

