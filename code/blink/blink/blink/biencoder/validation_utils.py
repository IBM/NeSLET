import json
from argparse import ArgumentError
import torch
import numpy as np
import os

from tqdm import tqdm, trange

from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

import blink.candidate_ranking.utils as utils


# This evaluate function during training uses in-batch negatives:
# for a batch of size B, the labels from the batch are used as label candidates
# B is controlled by the parameter eval_batch_size
def evaluate(
        reranker, eval_dataloader, params, device, logger,
):
    reranker.model.eval()
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        #         context_input, candidate_input, _, _ = batch
        context_input, candidate_input, _ = batch
        with torch.no_grad():
            eval_loss, logits = reranker(context_input, candidate_input)

        logits = logits.detach().cpu().numpy()
        # Using in-batch negatives, the label ids are diagonal
        label_ids = torch.LongTensor(torch.arange(params["eval_batch_size"])).numpy()
        tmp_eval_accuracy = utils.accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += context_input.size(0)
        nb_eval_steps += 1

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    results["normalized_accuracy"] = normalized_eval_accuracy
    return results


class Evaluator:
    def __init__(self, eval_dataloader, logger, params, device) -> None:
        self.logger = logger
        self.labels = None
        self.entity_descriptions_dataloader = None
        self.eval_dataloader = eval_dataloader
        self.num_entities = None
        self.params = params
        self.device = device

        self.build_entity_list_and_labels(self.eval_dataloader)

    def build_entity_list_and_labels(self, eval_dataloader):
        entity_description_list = []
        labels_list = []
        entity_description_to_id = {}

        for batch in eval_dataloader:
            _, candidate_input, _ = batch
            ent_descriptions = [tuple(row) for row in candidate_input.numpy()]
            for ent_desc in ent_descriptions:
                if ent_desc not in entity_description_to_id:
                    entity_description_to_id[ent_desc] = len(entity_description_to_id)
                    entity_description_list.append(ent_desc)

                labels_list.append(entity_description_to_id[ent_desc])

        self.num_entities = len(entity_description_to_id)

        assert len(entity_description_list) == len(entity_description_to_id)
        assert max(labels_list) < len(entity_description_list)

        self.logger.info(
            "Found {} unique entities in the val set".format(
                len(entity_description_list)
            )
        )

        entity_descriptions_tensor = torch.tensor(
            entity_description_list, dtype=torch.long
        )

        entity_descriptions_tensor_dataset = TensorDataset(entity_descriptions_tensor)

        valid_sampler = SequentialSampler(entity_descriptions_tensor_dataset)

        self.entity_descriptions_dataloader = DataLoader(
            entity_descriptions_tensor_dataset,
            sampler=valid_sampler,
            batch_size=self.params["eval_batch_size"],
        )

        self.labels = labels_list

    def compute_accuracy(self, y_pred, y_true):
        """
        y_pred: list
        y_true: list
        """
        assert len(y_pred) == len(y_true)

        num_correct = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                num_correct += 1

        accuracy = num_correct / len(y_pred)
        return accuracy

    def compute_entity_vectors(self, model):
        entity_vectors = []
        for batch in self.entity_descriptions_dataloader:
            with torch.no_grad():
                ent_vecs = model.encode_candidate(batch[0].to(self.device))
            entity_vectors.append(ent_vecs)

        entity_vectors = torch.cat(entity_vectors, dim=0)

        assert entity_vectors.shape[0] == self.num_entities
        return entity_vectors

    def get_predictions(self, model, entity_vectors):
        predictions = []
        for batch in tqdm(self.eval_dataloader, desc="Evaluation"):
            context_input, _, _ = batch
            with torch.no_grad():
                scores = model.score_candidate(
                    context_input.to(self.device),
                    None,
                    cand_encs=entity_vectors.to(self.device),
                )
                scores, indices = scores.topk(1)
                indices = indices.cpu()
            predictions.extend(indices.squeeze().tolist())

        return predictions

    def evaluate(self, model):
        self.logger.info("--------------- Started evaluation ---------------")

        model.model.eval()

        entity_vectors = self.compute_entity_vectors(model)
        predictions = self.get_predictions(model, entity_vectors)
        accuracy = self.compute_accuracy(y_pred=predictions, y_true=self.labels)

        results = {}
        self.logger.info("Eval accuracy: %.5f" % accuracy)
        results["normalized_accuracy"] = accuracy

        model.model.train()

        self.logger.info("--------------- Completed evaluation ---------------")

        return results


class Evaluator2:
    # Evaluates performance on entity linking and type prediction

    def __init__(self, eval_dataloader, logger, params, device) -> None:
        self.logger = logger
        self.entity_labels = None
        self.type_labels = None
        self.entity_descriptions_dataloader = None
        self.eval_dataloader = eval_dataloader
        self.num_entities = None
        self.params = params
        self.device = device

        self.build_entity_list_and_labels(self.eval_dataloader)

    def build_entity_list_and_labels(self, eval_dataloader):
        entity_description_list = []
        entity_labels_list = []
        entity_description_to_id = {}
        type_labels = []

        for batch in eval_dataloader:
            # _, candidate_input, _ = batch
            _, candidate_input, _, type_label_vec, _ = batch

            for t_lbl_vec in type_label_vec:
                positive_types = (t_lbl_vec == 1).nonzero().flatten().tolist()
                type_labels.append(positive_types)

            ent_descriptions = [tuple(row) for row in candidate_input.numpy()]
            for ent_desc in ent_descriptions:
                if ent_desc not in entity_description_to_id:
                    entity_description_to_id[ent_desc] = len(entity_description_to_id)
                    entity_description_list.append(ent_desc)

                entity_labels_list.append(entity_description_to_id[ent_desc])

        self.num_entities = len(entity_description_to_id)

        assert len(entity_description_list) == len(entity_description_to_id)
        assert max(entity_labels_list) < len(entity_description_list)

        self.logger.info(
            "Found {} unique entities in the val set".format(
                len(entity_description_list)
            )
        )

        entity_descriptions_tensor = torch.tensor(
            entity_description_list, dtype=torch.long
        )

        entity_descriptions_tensor_dataset = TensorDataset(entity_descriptions_tensor)

        valid_sampler = SequentialSampler(entity_descriptions_tensor_dataset)

        self.entity_descriptions_dataloader = DataLoader(
            entity_descriptions_tensor_dataset,
            sampler=valid_sampler,
            batch_size=self.params["eval_batch_size"],
        )

        self.entity_labels = entity_labels_list
        self.type_labels = type_labels

        assert len(self.entity_labels) == len(self.type_labels)

    def compute_accuracy(self, y_pred, y_true):
        """
        y_pred: list
        y_true: list
        """
        assert len(y_pred) == len(y_true)

        num_correct = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                num_correct += 1

        accuracy = num_correct / len(y_pred)
        return accuracy

    def compute_f1(self, y_pred, y_true):
        """
        y_pred: list of lists
        y_true: list of lists
        """
        assert len(y_pred) == len(y_true)

        total_f1 = 0
        total_precision = 0
        total_recall = 0
        for i in range(len(y_pred)):
            pred = set(y_pred[i])
            gold = set(y_true[i])

            if len(gold) == 0:
                if len(pred) == 0:
                    precision = 1
                    recall = 1
                else:
                    precision = 0
                    recall = 0
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

        count_empty_types = 0
        for gold_types in y_pred:
            if len(gold_types) == 0:
                count_empty_types += 1

        # assumption: P=R=F1=1 when gold types are empty. Hold true when we use the gold k to find topk before
        # computing the F1
        corrected_denominator = max(1, (len(y_pred) - count_empty_types))
        corrected_avg_f1 = (total_f1 - count_empty_types) / corrected_denominator
        corrected_avg_precision = (total_precision - count_empty_types) / corrected_denominator
        corrected_avg_recall = (total_recall - count_empty_types) / corrected_denominator

        avg_f1 = total_f1 / len(y_pred)
        avg_precision = total_precision / len(y_pred)
        avg_recall = total_recall / len(y_pred)

        return {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "corrected_avg_precision": corrected_avg_precision,
            "corrected_avg_recall": corrected_avg_recall,
            "corrected_avg_f1": corrected_avg_f1,
        }

    def compute_entity_vectors(self, model):
        entity_vectors = []
        for batch in self.entity_descriptions_dataloader:
            with torch.no_grad():
                ent_vecs = model.encode_candidate(batch[0].to(self.device))
            entity_vectors.append(ent_vecs)

        entity_vectors = torch.cat(entity_vectors, dim=0)

        assert entity_vectors.shape[0] == self.num_entities
        return entity_vectors

    def get_predictions(self, model, entity_vectors):
        entity_predictions = []
        type_predictions = []

        entity_prediction_scores = []
        type_prediction_scores = []

        for batch in tqdm(self.eval_dataloader, desc="Evaluation"):
            # context_input, candidate_input, label_ids, type_label_vec, ids = batch
            context_input, _, _, type_label_vec, _ = batch
            with torch.no_grad():
                entity_scores, type_scores = model.score_candidate(
                    text_vecs=context_input.to(self.device),
                    cand_vecs=None,
                    random_negs=True,
                    cand_encs=entity_vectors.to(
                        self.device
                    ),  # pre-computed candidate encoding.
                )

                entity_prediction_scores.extend(entity_scores.squeeze().tolist())
                type_prediction_scores.extend(torch.sigmoid(type_scores).squeeze().tolist())

                entity_scores, entity_indices = entity_scores.topk(1)
                entity_indices = entity_indices.cpu()
                entity_predictions.extend(entity_indices.squeeze().tolist())

                # type_probs = torch.sigmoid(type_scores)
                k_to_use_for_types = torch.sum(type_label_vec, dim=-1).to(torch.long)

                for i in range(len(type_scores)):
                    t_scores, pred_t_indices = type_scores[i].topk(
                        k_to_use_for_types[i].item()
                    )
                    pred_t_indices = pred_t_indices.cpu().tolist()
                    type_predictions.append(pred_t_indices)

        assert len(entity_predictions) == len(type_predictions) == \
               len(entity_prediction_scores) == len(type_prediction_scores)

        out_dict = {
            "entity_predictions": entity_predictions,
            "type_predictions": type_predictions,
            "entity_prediction_scores": entity_prediction_scores,
            "type_prediction_scores": type_prediction_scores
        }

        return out_dict

    def evaluate(self, model, dataset_path="", dump=False, evaluate_entities=True):
        self.logger.info("--------------- Started evaluation ---------------")

        model.model.eval()

        if evaluate_entities:
            entity_vectors = self.compute_entity_vectors(model)
        else:
            # fake entity vectors so that the rest of the code doesn't break
            entity_vectors = torch.ones([2, 768], dtype=torch.float32, device=self.device)

        predictions_all = self.get_predictions(
            model, entity_vectors
        )

        dataset_name = None
        if dump:
            assert dataset_path != ""
            dataset_name = os.path.basename(dataset_path)
            dump_file = os.path.join(self.params["output_path"],  "type_prediction_" + dataset_name)
            with open(dump_file, 'w') as out_file:
                for type_prob in predictions_all["type_prediction_scores"]:
                    json.dump(type_prob, out_file)
                    out_file.write("\n")

        entity_predictions = predictions_all["entity_predictions"]
        type_predictions = predictions_all["type_predictions"]

        if evaluate_entities:
            entity_accuracy = self.compute_accuracy(
                y_pred=entity_predictions, y_true=self.entity_labels
            )
        else:
            entity_accuracy = 0

        type_f1_metrics = self.compute_f1(
            y_pred=type_predictions, y_true=self.type_labels
        )

        if dataset_name is not None:
            self.logger.info("Dataset: {}".format(dataset_name))

        if evaluate_entities:
            self.logger.info("entity_accuracy: %.4f" % entity_accuracy)
        else:
            self.logger.info("Did not evaluate entity accuracy")


        self.logger.info(
            "Type P:{}, R:{}, F1:{}".format(
                type_f1_metrics["avg_precision"],
                type_f1_metrics["avg_recall"],
                type_f1_metrics["avg_f1"],
            )
        )

        self.logger.info(
            "Corrected Type P:{}, R:{}, F1:{}".format(
                type_f1_metrics["corrected_avg_precision"],
                type_f1_metrics["corrected_avg_recall"],
                type_f1_metrics["corrected_avg_f1"],
            )
        )

        results = {"corrected_avg_f1": type_f1_metrics["corrected_avg_f1"], "avg_f1": type_f1_metrics["avg_f1"]}

        # to make the training code use either entity or type accuracy to decide if we have a "new best model"
        # after each epoch
        if self.params["main_metric"] == "entity":
            results["normalized_accuracy"] = entity_accuracy
        elif self.params["main_metric"] == "type":
            results["normalized_accuracy"] = type_f1_metrics["corrected_avg_f1"]
        elif self.params["main_metric"] == "entity_and_type":
            raise NotImplementedError
        else:
            assert False, "Unsupported value for main_metric"

        # put the model to the training mode, in case that is not being done in the training code
        model.model.train()

        # self.logger.info("--------------- Completed evaluation ---------------")

        return results


class Evaluator3:
    # Evaluates performance on entity linking and type prediction

    def __init__(self, eval_dataloader, logger, params, device) -> None:
        self.logger = logger
        self.entity_labels = None
        self.type_labels = None
        self.entity_descriptions_dataloader = None
        self.eval_dataloader = eval_dataloader
        self.num_entities = None
        self.params = params
        self.device = device

        self.build_entity_list_and_labels(self.eval_dataloader)

    def build_entity_list_and_labels(self, eval_dataloader):
        entity_description_list = []
        entity_labels_list = []
        entity_description_to_id = {}
        type_labels = []

        for batch in eval_dataloader:
            _, candidate_input, _, type_label_vec, _ = batch

            type_label_vec = type_label_vec[:, 0, :]

            for t_lbl_vec in type_label_vec:
                positive_types = (t_lbl_vec == 1).nonzero().flatten().tolist()
                type_labels.append(positive_types)

            ent_descriptions = [tuple(row) for row in candidate_input.numpy()]
            for ent_desc in ent_descriptions:
                if ent_desc not in entity_description_to_id:
                    entity_description_to_id[ent_desc] = len(entity_description_to_id)
                    entity_description_list.append(ent_desc)

                entity_labels_list.append(entity_description_to_id[ent_desc])

        self.num_entities = len(entity_description_to_id)

        assert len(entity_description_list) == len(entity_description_to_id)
        assert max(entity_labels_list) < len(entity_description_list)

        self.logger.info(
            "Found {} unique entities in the val set".format(
                len(entity_description_list)
            )
        )

        entity_descriptions_tensor = torch.tensor(
            entity_description_list, dtype=torch.long
        )

        entity_descriptions_tensor_dataset = TensorDataset(entity_descriptions_tensor)

        valid_sampler = SequentialSampler(entity_descriptions_tensor_dataset)

        self.entity_descriptions_dataloader = DataLoader(
            entity_descriptions_tensor_dataset,
            sampler=valid_sampler,
            batch_size=self.params["eval_batch_size"],
        )

        self.entity_labels = entity_labels_list
        self.type_labels = type_labels

        assert len(self.entity_labels) == len(self.type_labels)

    def compute_accuracy(self, y_pred, y_true):
        """
        y_pred: list
        y_true: list
        """
        assert len(y_pred) == len(y_true)

        num_correct = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                num_correct += 1

        accuracy = num_correct / len(y_pred)
        return accuracy

    def compute_f1(self, y_pred, y_true):
        """
        y_pred: list of lists
        y_true: list of lists
        """
        assert len(y_pred) == len(y_true)

        total_f1 = 0
        total_precision = 0
        total_recall = 0
        for i in range(len(y_pred)):
            pred = set(y_pred[i])
            gold = set(y_true[i])

            if len(gold) == 0:
                if len(pred) == 0:
                    precision = 1
                    recall = 1
                else:
                    precision = 0
                    recall = 0
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

        count_empty_types = 0
        for gold_types in y_pred:
            if len(gold_types) == 0:
                count_empty_types += 1

        # assumption: P=R=F1=1 when gold types are empty. Hold true when we use the gold k to find topk before
        # computing the F1
        corrected_denominator = max(1, (len(y_pred) - count_empty_types))
        corrected_avg_f1 = (total_f1 - count_empty_types) / corrected_denominator
        corrected_avg_precision = (total_precision - count_empty_types) / corrected_denominator
        corrected_avg_recall = (total_recall - count_empty_types) / corrected_denominator

        avg_f1 = total_f1 / len(y_pred)
        avg_precision = total_precision / len(y_pred)
        avg_recall = total_recall / len(y_pred)

        return {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "corrected_avg_precision": corrected_avg_precision,
            "corrected_avg_recall": corrected_avg_recall,
            "corrected_avg_f1": corrected_avg_f1,
        }

    def compute_entity_vectors(self, model):
        entity_vectors = []
        for batch in self.entity_descriptions_dataloader:
            with torch.no_grad():
                ent_vecs = model.encode_candidate(batch[0].to(self.device))
            entity_vectors.append(ent_vecs)

        entity_vectors = torch.cat(entity_vectors, dim=0)

        assert entity_vectors.shape[0] == self.num_entities
        return entity_vectors

    def get_predictions(self, model, entity_vectors):
        entity_predictions = []
        type_predictions = []

        entity_prediction_scores = []
        type_prediction_scores = []

        for batch in tqdm(self.eval_dataloader, desc="Evaluation"):
            context_input, _, _, type_label_vec, _ = batch
            type_label_vec = type_label_vec[:, 0, :]

            with torch.no_grad():
                entity_scores, type_scores = model.forward_inference(
                    context_input=context_input.to(self.device),
                    cand_input=None,
                    cand_encs=entity_vectors.to(self.device)  # pre-computed candidate encoding
                )

                entity_prediction_scores.extend(entity_scores.squeeze().tolist())
                type_prediction_scores.extend(type_scores.squeeze().tolist())

                entity_scores, entity_indices = entity_scores.topk(1)
                entity_indices = entity_indices.cpu()
                entity_predictions.extend(entity_indices.squeeze().tolist())

                k_to_use_for_types = torch.sum(type_label_vec, dim=-1).to(torch.long)

                for i in range(len(type_scores)):
                    t_scores, pred_t_indices = type_scores[i].topk(
                        k_to_use_for_types[i].item()
                    )
                    pred_t_indices = pred_t_indices.cpu().tolist()
                    type_predictions.append(pred_t_indices)

        assert len(entity_predictions) == len(type_predictions) == \
               len(entity_prediction_scores) == len(type_prediction_scores)

        out_dict = {
            "entity_predictions": entity_predictions,
            "type_predictions": type_predictions,
            "entity_prediction_scores": entity_prediction_scores,
            "type_prediction_scores": type_prediction_scores
        }

        return out_dict

    def evaluate(self, model, dataset_path="", dump=False, evaluate_entities=True):
        self.logger.info("--------------- Started evaluation ---------------")

        model.model.eval()

        if evaluate_entities:
            entity_vectors = self.compute_entity_vectors(model)
        else:
            # fake entity vectors so that the rest of the code doesn't break
            entity_vectors = torch.ones([2, 768], dtype=torch.float32, device=self.device)

        predictions_all = self.get_predictions(
            model, entity_vectors
        )

        dataset_name = None
        if dump:
            assert dataset_path != ""
            dataset_name = os.path.basename(dataset_path)
            dump_file = os.path.join(self.params["output_path"],  "type_prediction_" + dataset_name)
            with open(dump_file, 'w') as out_file:
                for type_prob in predictions_all["type_prediction_scores"]:
                    json.dump(type_prob, out_file)
                    out_file.write("\n")

        entity_predictions = predictions_all["entity_predictions"]
        type_predictions = predictions_all["type_predictions"]

        if evaluate_entities:
            entity_accuracy = self.compute_accuracy(
                y_pred=entity_predictions, y_true=self.entity_labels
            )
        else:
            entity_accuracy = 0

        type_f1_metrics = self.compute_f1(
            y_pred=type_predictions, y_true=self.type_labels
        )

        if dataset_name is not None:
            self.logger.info("Dataset: {}".format(dataset_name))

        if evaluate_entities:
            self.logger.info("entity_accuracy: %.4f" % entity_accuracy)
        else:
            self.logger.info("Did not evaluate entity accuracy")


        self.logger.info(
            "Type P:{}, R:{}, F1:{}".format(
                type_f1_metrics["avg_precision"],
                type_f1_metrics["avg_recall"],
                type_f1_metrics["avg_f1"],
            )
        )

        self.logger.info(
            "Corrected Type P:{}, R:{}, F1:{}".format(
                type_f1_metrics["corrected_avg_precision"],
                type_f1_metrics["corrected_avg_recall"],
                type_f1_metrics["corrected_avg_f1"],
            )
        )

        results = {"corrected_avg_f1": type_f1_metrics["corrected_avg_f1"], "avg_f1": type_f1_metrics["avg_f1"]}

        # to make the training code use either entity or type accuracy to decide if we have a "new best model"
        # after each epoch
        if self.params["main_metric"] == "entity":
            results["normalized_accuracy"] = entity_accuracy
        elif self.params["main_metric"] == "type":
            results["normalized_accuracy"] = type_f1_metrics["corrected_avg_f1"]
        elif self.params["main_metric"] == "entity_and_type":
            raise NotImplementedError
        else:
            assert False, "Unsupported value for main_metric"

        # put the model to the training mode, in case that is not being done in the training code
        model.model.train()

        # self.logger.info("--------------- Completed evaluation ---------------")

        return results


def evaluator_factory(model_number, params, valid_dataloader, logger, device):
    if model_number == 1:
        return Evaluator(
            eval_dataloader=valid_dataloader,
            logger=logger,
            params=params,
            device=device,
        )
    elif model_number in [2, 4]:
        return Evaluator2(
            eval_dataloader=valid_dataloader,
            logger=logger,
            params=params,
            device=device,
        )
    elif model_number in [3, 5]:
        return Evaluator3(
            eval_dataloader=valid_dataloader,
            logger=logger,
            params=params,
            device=device,
        )
    else:
        assert False, "Unsupported model_number"
