import argparse
import numpy as np
from scipy.special import softmax
from collections import defaultdict
import json
import options
import copy

with open(options.cached_title2id) as fp:
    title2id = json.load(fp)

with open(options.cached_id2title) as fp:
    id2title = json.load(fp)

id2title = {int(k): v for k, v in id2title.items()}

with open(options.cached_wikipedia_id2local_id) as fp:
    wikipedia_id2local_id = json.load(fp)

wikipedia_id2local_id = {int(k): v for k, v in wikipedia_id2local_id.items()}

with open(options.cached_id2text) as fp:
    id2text = json.load(fp)

id2text = {int(k): v for k, v in id2text.items()}

with open(options.cached_id_to_type_dict_path) as f:
    id_to_type_dict = json.load(f)
id_to_type_dict = {int(k): v for k, v in id_to_type_dict.items()}

with open(options.cached_id_to_fine_type_dict_path) as f:
    id_to_fine_type_dict = json.load(f)
id_to_fine_type_dict = {int(k): v for k, v in id_to_fine_type_dict.items()}

with open(options.ontology_nodes_parent_file_path) as f:
    parent_of_node = json.load(f)

with open(options.type_co_occurrence_prob_file_path) as f:
  co_occurrence_matrix = json.load(f)
  co_occurrence_matrix = {int(k): v for k, v in co_occurrence_matrix.items()}

with open(options.type_dict_path) as f:
  id_to_type = json.load(f)

root_node_index = len(id_to_type)



def get_fine_types(all_types,parent_of_node):
    fine_types = []
    for type_consider in all_types:
        is_parent  = False
        for type_check in all_types:
            if type_consider == type_check:
                continue
            path_to_root = get_leaf_to_root_path(type_check,parent_of_node)
            if type_consider in path_to_root:
                is_parent = True
                break
        if is_parent == False:
            fine_types.append(type_consider)

    return fine_types

def calculate_indices(score,old_indicies):
    score = np.array(score)
    old_indicies = np.array(old_indicies)
    new_indicies = (-score).argsort(kind='mergesort')
    new_score = score[new_indicies]
    new_indicies = old_indicies[new_indicies]
    return np.reshape(new_indicies, (1, new_indicies.size)) ,new_score

def prune_entity_types(types,max_level):
    new_types = []
    for type in types:
        path = get_leaf_to_root_path(type,parent_of_node)
        if len(path) > max_level + 1:
            l = len(path)
            new_path = path[l-max_level-1:]
            new_types.append(new_path[0])
        else:
            new_types.append(type)

    return new_types

def get_leaf_nodes(parent_of_node):
    leaf_nodes = []

    for index,node in enumerate(parent_of_node):
        if parent_of_node.count(index) == 0:
            leaf_nodes.append(index)

    return leaf_nodes

def get_leaf_to_root_path(node,parent_of_node):
    path = []

    start = node
    while (start != -1):
        path.append(start)
        start = parent_of_node[start]

    path.append(root_node_index)
    return path

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def get_entity_probability(entity_path,parent_of_node,co_occurrence_matrix):
    new_entity_path = copy.deepcopy(entity_path)
    if entity_path['full_path'] == True:
        prob = 1
    else:
        path = entity_path['path']
        prob = 1
        for index,node in enumerate(path):
            if index == entity_path['common_node_index']:
                break
            parent  = path[index+1]
            child = path[index]
            prob = prob * co_occurrence_matrix[parent][child]
            #print(co_occurrence_matrix[parent][child])
    new_entity_path['prob'] = prob
    return new_entity_path

def get_entity_paths_in_ontology(entity_fine_types,leaf_to_root_paths_in_ontology,max_level):
    paths = {}
    entity_fine_types = prune_entity_types(entity_fine_types,max_level)
    if len(entity_fine_types) > 0:
        for node in leaf_to_root_paths_in_ontology:
           common_nodes = intersection(entity_fine_types, leaf_to_root_paths_in_ontology[node])
           if len(common_nodes) > 0:
               assert len(common_nodes) == 1
               temp_path = {}
               temp_path['path'] = leaf_to_root_paths_in_ontology[node]
               if leaf_to_root_paths_in_ontology[node][0] in entity_fine_types:
                   temp_path['full_path'] = True
                   temp_path['common_node'] = leaf_to_root_paths_in_ontology[node][0]
                   temp_path['common_node_index'] = 0
               else:
                   temp_path['full_path'] = False
                   temp_path['common_node'] = common_nodes[0]
                   temp_path['common_node_index'] = temp_path['path'].index(common_nodes[0])


               paths[node]=temp_path
    else:
        for node in leaf_to_root_paths_in_ontology:
            temp_path = {}
            temp_path['path'] = leaf_to_root_paths_in_ontology[node]
            temp_path['full_path'] = False
            temp_path['common_node'] = root_node_index
            temp_path['common_node_index'] = temp_path['path'].index(root_node_index)
            paths[node] = temp_path

    return paths

def get_mention_prrobablity(all_types_probablity,path):
    return all_types_probablity[path[0]]

def get_path_mention_probablity(leaf_to_root_paths_in_ontology,all_types_probablity):
    path_mention_probablity = {}
    for node in leaf_to_root_paths_in_ontology:
        temp_path = {}
        temp_path['path'] = leaf_to_root_paths_in_ontology[node]
        prob = get_mention_prrobablity(all_types_probablity,temp_path['path'])
        temp_path['prob'] = prob
        path_mention_probablity[node] = temp_path
    return path_mention_probablity

leaf_nodes_ontology = get_leaf_nodes(parent_of_node)
leaf_to_root_paths_in_ontology = {}

def get_leaf_to_root_paths_in_ontology(max_level : int) -> dict:
    max_level_in_ontology = 0
    max_path_in_ontology = []
    for node in leaf_nodes_ontology:
        path  = get_leaf_to_root_path(node,parent_of_node)
        if max_level_in_ontology < len(path):
            max_level_in_ontology = len(path)-1
            max_path_in_ontology = path
        if len(path) <= max_level + 1:
            leaf_to_root_paths_in_ontology[node] = path
        else:
            l = len(path)
            new_path = path[l - max_level - 1:]
            leaf_to_root_paths_in_ontology[new_path[0]] = new_path
    return leaf_to_root_paths_in_ontology


# perform_type_based_inference function is function to perform type based inference
# top_k_entities is a list of top_k entities returned by BLINK
# top_k_entities_scores a list of scores of top_k entities returned by BLINK
# type_probablities is a list of type probabilities
# levels_to_prune is of type int and used to prune ontology tree at a level

def perform_type_based_inference(top_k_entities: list, top_k_entities_scores: list, type_probablities: list, levels_to_prune: int = 3) -> dict:
    assert len(id_to_type) == len(type_probablities)
    assert len(top_k_entities) == len(top_k_entities_scores)

    if levels_to_prune > options.max_tree_depth:
        levels_to_prune = options.max_tree_depth

    result = {}


    labels = []
    nns_type = []
    nns_both = []
    nns_blink = []
    nns_multiply = []

    top_k = len(top_k_entities)

    leaf_to_root_paths_in_ontology = get_leaf_to_root_paths_in_ontology (levels_to_prune)


        # print(id_to_type_probablity[id])

        #type_prediction['id'] = id
        #sample = test_samples[id]
        #label_id = wikipedia_id2local_id[int(sample['label_id'])]


        #labels.extend([label_id])




    all_types_probablity = type_probablities
    path_mention_probablity = get_path_mention_probablity(leaf_to_root_paths_in_ontology, all_types_probablity)
    top_blink_predictions_for_mention = [title2id[item] for item in top_k_entities]
    blink_scores = top_k_entities_scores[:top_k]
    type_based_scores = []
    for entity_id in top_blink_predictions_for_mention[:top_k]:
        # print(entity_id)
        # print(id2title[entity_id])
        # print(id_to_type_dict[entity_id])
        # print([id_to_type[str(item)] for item in id_to_type_dict[entity_id]])

        entity_fine_types = id_to_fine_type_dict[entity_id]

        entity_paths_in_ontology = get_entity_paths_in_ontology(entity_fine_types,
                                                                leaf_to_root_paths_in_ontology,levels_to_prune)
        # print(len(entity_paths_in_ontology))

        type_score = 0
        for node in entity_paths_in_ontology:
            new_entity_path = get_entity_probability(entity_paths_in_ontology[node], parent_of_node,
                                                     co_occurrence_matrix)
            type_score += new_entity_path['prob'] * path_mention_probablity[node]['prob']
        # print(type_score)
        type_based_scores.append(type_score)
    type_based_scores = softmax(type_based_scores)
    blink_scores = softmax(blink_scores)

    total_scores_both = blink_scores + type_based_scores
    total_scores_type = type_based_scores
    total_scores_blink = blink_scores
    total_score_multiply = np.multiply(blink_scores, (1 + type_based_scores))

    new_indices_blink, new_blink_score = calculate_indices(total_scores_blink, top_blink_predictions_for_mention)

    new_indices_type, new_type_score = calculate_indices(total_scores_type, top_blink_predictions_for_mention)

    new_indices_both, new_both_score = calculate_indices(total_scores_both, top_blink_predictions_for_mention)

    new_indices_multiply,new_multiply_score = calculate_indices(total_score_multiply, top_blink_predictions_for_mention)

    #result['Prediction_Blink'] = [id2title[item] for item in new_indices_blink[0]]
    #result['Prediction_Type'] = [id2title[item] for item in new_indices_type[0]]
    #result['Prediction_Blink_Plus_Type'] = [id2title[item] for item in new_indices_both[0]]
    #result['Prediction_Multiply'] = [id2title[item] for item in new_indices_multiply[0]]
    #result['Score_Blink'] = new_blink_score.tolist()
    #result['Score_Type'] = new_type_score.tolist()
    #result['Score_Blink_Plus_Type'] = new_both_score.tolist()
    #result['Score_Prediction_Multiply'] = new_multiply_score.tolist()

    result['Score_Type'] = type_based_scores.tolist()

    return result