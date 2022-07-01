import torch
import json

import argparse

def read_jsonl(file_path):
    data_out = []
    with open(file_path) as in_file:
        for line in in_file:
            data_out.append(json.loads(line))
    return data_out


def jsonl_to_tensor(file_path):
    data_out = []
    with open(file_path) as in_file:
        for line in in_file:
            data_out.append(json.loads(line))
    return torch.tensor(data_out)

def read_json(file_path):
    with open(file_path) as in_file:
        data = json.load(in_file)
    return data


def write_jsonl(data_in, file_path):
    with open(file_path, 'w') as out_file:
        for item in data_in:
            json.dump(item, out_file)
            out_file.write("\n")


def apply_threshold(predicted_probabilities, threshold):
    binarized_predictions = (predicted_probabilities > threshold).to(torch.int32)

    predicted_ids = []
    for i, bin_pred in enumerate(binarized_predictions):
        pred_ids = (bin_pred == 1).nonzero().flatten().tolist()

        if not any(pred_ids):
            # take top1 predicted type so that the predicted_ids[i] is never empty
            _, pred_ids = predicted_probabilities[i].topk(1)
            pred_ids = pred_ids.tolist()

        predicted_ids.append(pred_ids)

    assert len(predicted_ids) == predicted_probabilities.shape[0]

    return predicted_ids


# def convert_id_to_name(predicted_ids, id_to_name):
#     predicted_names = []
#     for pred_ids in predicted_ids:
#         predicted_names.append([id_to_name[str(i)] for i in pred_ids])
#
#     return predicted_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted_probabilities", type=str)
    parser.add_argument("--file_to_fill", type=str)
    parser.add_argument("--output_file", type=str)
    # parser.add_argument("--type_id_to_name", type=str)
    parser.add_argument("--threshold", type=float)

    config = parser.parse_args()

    predicted_probabilities = jsonl_to_tensor(config.predicted_probabilities)
    # id_to_name = read_json(config.type_id_to_name)

    predicted_type_ids = apply_threshold(predicted_probabilities, config.threshold)

    # predicted_types = convert_id_to_name(predicted_ids=predicted_type_ids, id_to_name=id_to_name)

    entity_file_to_fill = read_jsonl(config.file_to_fill)

    assert len(entity_file_to_fill) == len(predicted_type_ids)

    for i, data in enumerate(entity_file_to_fill):
        data["predicted_types"] = predicted_type_ids[i]

    write_jsonl(entity_file_to_fill, config.output_file)

    print("Done. Output written to {}".format(config.output_file))





