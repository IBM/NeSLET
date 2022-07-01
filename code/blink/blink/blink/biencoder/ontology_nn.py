import torch
import torch.nn as nn
from collections import defaultdict
import json


class OntologyNN(nn.Module):
    def __init__(self, config):
        super(OntologyNN, self).__init__()
        self.config = config
        self.aggregation_function = self.get_aggregation_function()
        self.child_to_parent = self.read_child_to_parent_from_file(self.config["ontology_file"])
        self.parent_to_child = self.build_parent_to_child(self.child_to_parent)
        self.set_of_leaves = self.get_leaf_nodes(self.child_to_parent)
        self.node_processing_order = self.get_node_processing_order(self.child_to_parent, self.set_of_leaves)

        if self.config["prob_aggregation"] == "weighted_lukasiewicz":
            self.weights_lukasiewicz = nn.Parameter(torch.randn(self.config["num_types"]))

    def read_child_to_parent_from_file(self, file_path):
        with open(file_path) as in_file:
            child_to_parent = json.load(in_file)
        return child_to_parent

    def build_parent_to_child(self, child_to_parent):
        """
        Returns a dictionary of the form
        {parent_node_idx: torch.tensor([child_node_1_idx, ..., child_node_n_idx]), ... }
        """
        parent_to_child = defaultdict(list)
        for child_idx, parent_idx in enumerate(child_to_parent):
            if parent_idx != -1:
                parent_to_child[parent_idx].append(child_idx)

        parent_to_child = {k: torch.tensor(v) for k, v in parent_to_child.items()}

        return parent_to_child

    def get_leaf_nodes(self, child_to_parent):
        set_of_nodes = set(range(len(child_to_parent)))
        set_of_parents = set(child_to_parent)
        set_of_leaves = set_of_nodes - set_of_parents
        return set_of_leaves

    def find_distance_to_root(self, node_idx):
        """
        This function returns the distance from node_idx to it's root. This distance may not be
        the path length but it varies proportionally to the path length. Good enough for our purpose.
        """
        if node_idx < 0:
            return 0

        distance = 1
        check_parent_idx = node_idx
        while self.child_to_parent[check_parent_idx] != -1:
            distance += 1
            check_parent_idx = self.child_to_parent[check_parent_idx]

        return distance

    def get_node_processing_order(self, child_to_parent, set_of_leaves):
        """
        Return an ordering of the non leaf nodes such that parent comes after child.
        """
        non_leaf_distance_to_root = []
        for node_idx in range(len(child_to_parent)):
            if node_idx not in set_of_leaves:
                node_distance_to_root = self.find_distance_to_root(node_idx)
                non_leaf_distance_to_root.append({"node_idx": node_idx, "distance": node_distance_to_root})

        non_leaf_distance_to_root_sorted = sorted(non_leaf_distance_to_root, reverse=True, key=lambda x: x["distance"])
        node_processing_order = [item["node_idx"] for item in non_leaf_distance_to_root_sorted]

        return node_processing_order

    def godel(self, predicted_probabilities_transpose, node_to_process):
        """
        p(parent) = max(p(child_1), ..., p(child_n))

        This function returns max(p(child_1), ..., p(child_n))
        """
        child_node_indices = self.parent_to_child[node_to_process]
        pred_prob_children_transpose = predicted_probabilities_transpose[child_node_indices]

        values, indices = pred_prob_children_transpose.max(dim=0)

        return values

    def lukasiewicz(self, predicted_probabilities_transpose, node_to_process):
        """
        p(parent) = min(1, p(child_1) +, ..., + p(child_n))

        This function returns min(1, p(child_1) +, ..., + p(child_n))
        """
        child_node_indices = self.parent_to_child[node_to_process]
        pred_prob_children_transpose = predicted_probabilities_transpose[child_node_indices]

        values = pred_prob_children_transpose.sum(dim=0)

        values_clamped = torch.clamp(values, min=0, max=1)

        return values_clamped

    def weighted_lukasiewicz(self, predicted_probabilities_transpose, node_to_process):
        """
        p(parent) = min(1, w1*p(child_1) +, ..., + wn*p(child_n))

        This function returns min(1, w1*p(child_1) +, ..., + wn*p(child_n))
        """
        child_node_indices = self.parent_to_child[node_to_process]
        pred_prob_children_transpose = predicted_probabilities_transpose[child_node_indices]

        pred_prob_children_transpose_weighted = pred_prob_children_transpose * torch.sigmoid(self.weights_lukasiewicz[child_node_indices].unsqueeze(0).permute(1,0))

        values = pred_prob_children_transpose_weighted.sum(dim=0)

        values_clamped = torch.clamp(values, min=0, max=1)

        return values_clamped

    def get_aggregation_function(self):
        if self.config["prob_aggregation"] == "godel":
            return self.godel
        elif self.config["prob_aggregation"] == "lukasiewicz":
            return self.lukasiewicz
        elif self.config["prob_aggregation"] == "weighted_lukasiewicz":
            return self.weighted_lukasiewicz
        else:
            raise ValueError("Unsupported value of prob_aggregation")

    def forward(self, predicted_probabilities):
        """
        predicted_probabilities: batch_size x num_nodes_in_ontology
        """
        assert len(predicted_probabilities.shape) == 2

        predicted_probabilities_transpose = predicted_probabilities.permute(1, 0)

        for node_to_process in self.node_processing_order:
            replacement = self.aggregation_function(
                predicted_probabilities_transpose,
                node_to_process)
            predicted_probabilities_transpose = predicted_probabilities_transpose.clone()
            predicted_probabilities_transpose[node_to_process] = replacement

        new_probabilities = predicted_probabilities_transpose.permute(1, 0)

        return new_probabilities




def test_godel():
    # This block serves as a test for OntologyNN with the Godel norm
    # ontology.json has [-1,0,1,1,2,2,2,3,-1,8,8,8,10,10]

    config = {
        "ontology_file": "NeSLET_everything/projects/ontology.json",
        "prob_aggregation": "godel"
    }

    ontology_nn = OntologyNN(config)

    ontology_nn.train()

    predicted_probabilities = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.5, 0.3, 0.7, 0.9, 0.2, 0.1],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.5, 0.7, 0.7, 0.1, 0.8, 0.9]
    ], requires_grad=True)

    expected_new_probabilities = torch.tensor([
        [0.8, 0.8, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8, 0.9, 0.3, 0.2, 0.9, 0.2, 0.1],
        [0.5, 0.5, 0.5, 0.2, 0.5, 0.4, 0.3, 0.2, 0.9, 0.7, 0.9, 0.1, 0.8, 0.9]
    ])

    new_probabilities = ontology_nn(predicted_probabilities)

    print("child_to_parent: ", ontology_nn.child_to_parent)
    print("======================================================")
    print("parent_to_child: ", ontology_nn.parent_to_child)
    print("======================================================")
    print("set_of_leaves: ", ontology_nn.set_of_leaves)
    print("======================================================")
    print("node_processing_order: ", ontology_nn.node_processing_order)
    print("======================================================")
    print("New computed probabilities: ", new_probabilities)
    print("======================================================")
    print("Expected new probabilities: ", expected_new_probabilities)
    print("======================================================")
    if torch.all(new_probabilities == expected_new_probabilities).item():
        print("new_probabilities = expected_new_probabilities. Test passed")
    else:
        print("new_probabilities != expected_new_probabilities. Test Failed")


def test_lukasiewicz():
    # This block serves as a test for OntologyNN with the lukasiewicz norm
    # ontology.json has [-1,0,1,1,2,2,2,3,-1,8,8,8,10,10]
    # right now, this function can't catch logic errors

    config = {
        "ontology_file": "NeSLET_everything/projects/ontology.json",
        "prob_aggregation": "lukasiewicz"
    }

    ontology_nn = OntologyNN(config)

    ontology_nn.train()

    predicted_probabilities = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.5, 0.3, 0.7, 0.9, 0.2, 0.1],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.5, 0.7, 0.7, 0.1, 0.8, 0.9]
    ], requires_grad=True)

    # expected_new_probabilities = torch.tensor([
    #     [0.8, 0.8, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8, 0.9, 0.3, 0.2, 0.9, 0.2, 0.1],
    #     [0.5, 0.5, 0.5, 0.2, 0.5, 0.4, 0.3, 0.2, 0.9, 0.7, 0.9, 0.1, 0.8, 0.9]
    # ])

    new_probabilities = ontology_nn(predicted_probabilities)

    print("child_to_parent: ", ontology_nn.child_to_parent)
    print("======================================================")
    print("parent_to_child: ", ontology_nn.parent_to_child)
    print("======================================================")
    print("set_of_leaves: ", ontology_nn.set_of_leaves)
    print("======================================================")
    print("node_processing_order: ", ontology_nn.node_processing_order)
    print("======================================================")
    print("New computed probabilities: ", new_probabilities)
    print("======================================================")
    # print("Expected new probabilities: ", expected_new_probabilities)
    print("Expected new probabilities: Not given")
    print("======================================================")
    # if torch.all(new_probabilities == expected_new_probabilities).item():
    #     print("new_probabilities = expected_new_probabilities. Test passed")
    # else:
    #     print("new_probabilities != expected_new_probabilities. Test Failed")


def test_weighted_lukasiewicz():
    # This block serves as a test for OntologyNN with the weighted lukasiewicz norm
    # ontology.json has [-1,0,1,1,2,2,2,3,-1,8,8,8,10,10]
    # right now, this function can't catch logic errors

    config = {
        "ontology_file": "NeSLET_everything/projects/ontology.json",
        "prob_aggregation": "weighted_lukasiewicz",
        "num_types": 14
    }

    ontology_nn = OntologyNN(config)

    ontology_nn.train()

    predicted_probabilities = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.5, 0.3, 0.7, 0.9, 0.2, 0.1],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.5, 0.7, 0.7, 0.1, 0.8, 0.9]
    ], requires_grad=True)

    # expected_new_probabilities = torch.tensor([
    #     [0.8, 0.8, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8, 0.9, 0.3, 0.2, 0.9, 0.2, 0.1],
    #     [0.5, 0.5, 0.5, 0.2, 0.5, 0.4, 0.3, 0.2, 0.9, 0.7, 0.9, 0.1, 0.8, 0.9]
    # ])

    new_probabilities = ontology_nn(predicted_probabilities)

    print("child_to_parent: ", ontology_nn.child_to_parent)
    print("======================================================")
    print("parent_to_child: ", ontology_nn.parent_to_child)
    print("======================================================")
    print("set_of_leaves: ", ontology_nn.set_of_leaves)
    print("======================================================")
    print("node_processing_order: ", ontology_nn.node_processing_order)
    print("======================================================")
    print("New computed probabilities: ", new_probabilities)
    print("======================================================")
    # print("Expected new probabilities: ", expected_new_probabilities)
    print("Expected new probabilities: Not given")
    print("======================================================")
    # if torch.all(new_probabilities == expected_new_probabilities).item():
    #     print("new_probabilities = expected_new_probabilities. Test passed")
    # else:
    #     print("new_probabilities != expected_new_probabilities. Test Failed")


if __name__ == "__main__":
    # test_godel()
    # test_lukasiewicz()
    test_weighted_lukasiewicz()






