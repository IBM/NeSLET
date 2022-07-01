import argparse
from threshold_type_probs import read_jsonl, jsonl_to_tensor, apply_threshold
import torch

def compute_f1(y_pred, y_true):
    """
    y_pred: list of lists
    y_true: list of lists

    ignores the data point if the gold list is empty
    """
    assert len(y_pred) == len(y_true)

    total_f1 = 0
    total_precision = 0
    total_recall = 0
    empty_counter = 0
    for i in range(len(y_pred)):
        pred = set(y_pred[i])
        gold = set(y_true[i])

        if len(gold) == 0:
            # if len(pred) == 0:
            #     precision = 1
            #     recall = 1
            # else:
            #     precision = 0
            #     recall = 0
            empty_counter += 1
            continue
        else:
            if len(pred) == 0:
                precision = 1
            else:
                precision = len(pred.intersection(gold)) / len(pred)

            recall = len(pred.intersection(gold)) / len(gold)

        total_precision += precision
        total_recall += recall

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = (2 * precision * recall) / (precision + recall)

        total_f1 += f1

    avg_f1 = total_f1 / (len(y_pred) - empty_counter)
    avg_precision = total_precision / (len(y_pred) - empty_counter)
    avg_recall = total_recall / (len(y_pred) - empty_counter)

    return {
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
    }


def get_gold_types(data):
    gold_types = []
    for d in data:
        gt = d["fine_types_id"] + d["coarse_types_id"] + d["ancestors_types_id"]
        gold_types.append(gt)
    return gold_types


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted_probabilities", type=str)
    parser.add_argument("--val_set", type=str)
    config = parser.parse_args()

    val_set = read_jsonl(config.val_set)
    gold_types = get_gold_types(data=val_set)

    predicted_probabilities = jsonl_to_tensor(config.predicted_probabilities)

    assert len(gold_types) == len(predicted_probabilities)

    thresholds = torch.arange(0, 1, 0.05)
    best_f1 = -1
    best_threshold = -1

    for t in thresholds:
        predicted_types = apply_threshold(predicted_probabilities, t)
        assert len(gold_types) == len(predicted_types)
        metrics = compute_f1(y_pred=predicted_types, y_true=gold_types)

        if metrics["avg_f1"] > best_f1:
            best_f1 = metrics["avg_f1"]
            best_threshold = t

    print("Best F1: {} at threshold {}".format(best_f1, best_threshold))


