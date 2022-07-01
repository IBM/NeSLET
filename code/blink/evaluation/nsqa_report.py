import json
import argparse


def read_first_k_lines(file_path, k=-1):
    data = []
    with open(file_path) as in_file:
        if k > 0:
            for i in range(k):
                data.append(json.loads(in_file.readline()))
        else:
            for line in in_file:
                data.append(json.loads(line))
    return data


def read_json(file_path):
    with open(file_path) as in_file:
        data = json.load(in_file)
    return data


def compute_mention_level_ems(pred, gold, topk):
    # pred: [p1, p2, p3, ...]
    # gold: [[g11, g12, g13, ...], [g21, g22, g23, ...], [g31, g32, g33, ...], ...]

    assert len(pred) == len(gold)
    assert topk > 0

    metrics = {"em": [], "recall_at_k": []}

    for i in range(len(pred)):
        if gold[i] == pred[i][0]:
            em = 1
        else:
            em = 0

        if gold[i] in pred[i][:topk]:
            recall_at_k = 1
        else:
            recall_at_k = 0

        metrics["em"].append(em)
        metrics["recall_at_k"].append(recall_at_k)

    assert len(metrics["em"]) == len(metrics["recall_at_k"]) == len(pred)

    return metrics


def make_gold_list(dataset):
    gold_entities = []
    for data_item in dataset:
        gold_entities.append(data_item["label_title"])

    return gold_entities


def merge_dataset_and_metrics(dataset, mention_level_ems):
    # modifies dataset in-place
    for i, d in enumerate(dataset):
        d["em"] = mention_level_ems["em"][i]
        d["recall_at_k"] = mention_level_ems["recall_at_k"][i]


def make_question_level_metrics(dataset_with_metrics):
    """
    dataset_with_metrics should have only one dataset
    """

    # find out the number of questions
    question_ids = set([])
    for d in dataset_with_metrics:
        question_ids.add(d["meta_question_id"])

    qids_and_ems = {i: 1 for i in question_ids}
    qids_and_recall_at_k = {i: 1 for i in question_ids}
    for d_idx, d in enumerate(dataset_with_metrics):
        qids_and_ems[d["meta_question_id"]] = qids_and_ems[d["meta_question_id"]] and d["em"]
        qids_and_recall_at_k[d["meta_question_id"]] = qids_and_recall_at_k[d["meta_question_id"]] and d["recall_at_k"]

    metrics = {"question_level_em": qids_and_ems.values(), "question_level_recall_at_k": qids_and_recall_at_k.values()}

    return metrics


def segregate_datasets(dataset):
    dataset_names = set([])
    for d in dataset:
        dataset_names.add(d["dataset_name"])

    segregated_data = {d_name: [] for d_name in dataset_names}
    for d in dataset:
        segregated_data[d["dataset_name"]].append(d)

    return segregated_data


def find_avg(list_in):
    return sum(list_in)/len(list_in)


def print_report(dataset, predictions, topk):
    gold = make_gold_list(dataset=dataset)
    mention_level_ems = compute_mention_level_ems(pred=predictions, gold=gold, topk=topk)

    overall_mention_level_em = sum(mention_level_ems["em"])/len(mention_level_ems["em"])
    overall_mention_level_recall_at_k = sum(mention_level_ems["recall_at_k"])/len(mention_level_ems["recall_at_k"])
    print("Overall mention level EM: {}, recall at {}: {}".format(overall_mention_level_em, topk, overall_mention_level_recall_at_k))

    # modifies dataset inplace
    # renaming the object for readability
    dataset_with_metrics = dataset
    merge_dataset_and_metrics(dataset=dataset_with_metrics, mention_level_ems=mention_level_ems)

    segregated_datasets = segregate_datasets(dataset_with_metrics)

    dataset_vs_average_mention_level_metrics = {}
    for dataset_name in segregated_datasets.keys():
        avg_em = find_avg([item["em"] for item in segregated_datasets[dataset_name]])
        avg_recall_at_k = find_avg([item["recall_at_k"] for item in segregated_datasets[dataset_name]])
        dataset_vs_average_mention_level_metrics[dataset_name] = {"avg_mention_level_em": avg_em, "avg_mention_level_recall_at_k": avg_recall_at_k}

    print("Mention level metrics for each dataset: {}".format(dataset_vs_average_mention_level_metrics))

    dataset_vs_average_question_level_metrics = {}
    total_num_questions_correct_at_1 = 0
    total_num_questions_correct_at_k = 0
    total_num_questions = 0
    for dataset_name in segregated_datasets.keys():
        q_level_metrics = make_question_level_metrics(dataset_with_metrics=segregated_datasets[dataset_name])

        total_num_questions_correct_at_1 += sum(q_level_metrics["question_level_em"])
        total_num_questions_correct_at_k += sum(q_level_metrics["question_level_recall_at_k"])
        total_num_questions += len(q_level_metrics["question_level_em"])

        avg_question_level_em = find_avg(q_level_metrics["question_level_em"])
        avg_question_level_recall_at_k = find_avg(q_level_metrics["question_level_recall_at_k"])

        dataset_vs_average_question_level_metrics[dataset_name] = {"avg_question_level_em": avg_question_level_em, "avg_question_level_recall_at_k": avg_question_level_recall_at_k}

    overall_question_level_em = total_num_questions_correct_at_1/total_num_questions
    overall_question_level_recall_at_k = total_num_questions_correct_at_k/total_num_questions
    print("Overall question level EM: {}, recall at {}: {}".format(overall_question_level_em, topk, overall_question_level_recall_at_k))

    print("Question level metrics for each dataset: {}".format(dataset_vs_average_question_level_metrics))


if __name__ == "__main__":
    # python nsqa_report.py --dataset /predictions/valid.jsonl --predictions /predictions/prediction.json --topk 5

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--predictions', type=str)
    parser.add_argument('--topk', type=int)
    config = parser.parse_args()
    config = config.__dict__

    dataset = read_first_k_lines(config["dataset"])
    predictions = read_json(config["predictions"])

    print_report(dataset, predictions, config["topk"])
