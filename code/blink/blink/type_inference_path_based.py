import json
import jsonlines
import argparse
from tqdm import tqdm, trange
import numpy as np
from scipy.special import softmax
from collections import defaultdict
import blink.candidate_ranking.utils as utils
from tqdm import tqdm, trange
import os
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--topk_file_path",type=str,help="Top-K prediction file path")
parser.add_argument("--entity_catalogue",dest="entity_catalogue",type=str,help="Path to the entity catalogue.")
parser.add_argument("--output_path", type=str, help="Path to the output.")
parser.add_argument("--debug",action="store_true",help="Whether to run in debug mode with only 200 samples.")
parser.add_argument("--type_probability_file_path",type=str,help="Type probability file path")
parser.add_argument("--type_co_occurrence_prob_file_path",type=str,help="Type probability file path")
parser.add_argument("--ontology_nodes_parent_file_path",type=str,help="Path to File which contains parent of every node in the ontology")
parser.add_argument("--type_dict_path",type=str, help="The path to the type dictionaries")
parser.add_argument("--ancestors", type=bool, default=False, help="Whether to use ancestor types in case of DBpedia types")
parser.add_argument("--entity_to_type_file_path",type=str, help="The path to the entity to type file")
parser.add_argument("--types_key",type=str, help="Type Key")
parser.add_argument("--cached_id_to_type_dict_path", type=str, help="Path to the cached id_to_type_dict for debug mode")
parser.add_argument("--cached_id_to_fine_type_dict_path", type=str, help="Path to the cached id_to_fine_type_dict for debug mode")
parser.add_argument("--data_path",type=str, help="The path to the test data.")
#parser.add_argument("--top_k", default=10, type=int)
parser.add_argument("--levels_to_prune", default=7, type=int)



args = parser.parse_args()
args.output_path = os.path.join(args.output_path,"level_"+str(args.levels_to_prune))
logger = utils.get_logger(args.output_path)
max_level = args.levels_to_prune


test_samples = {}
with open(args.data_path, "r") as fin:
    lines = fin.readlines()
    for line in lines:
        record = json.loads(line)
        if type(record['id']) == str:
            record['id'] = int(record['id'].replace("_", ""))

        if record['id'] in test_samples:
            record['id'] = int(str(99)+str(record['id']))
            test_samples[record['id']] = record
            #print("all ready there")
        else:
            test_samples[record['id']]=record

with open(args.type_dict_path) as f:
  id_to_type = json.load(f)

root_node_index = len(id_to_type)

def calculate_indices(score,old_indicies):
    score = np.array(score)
    old_indicies = np.array(old_indicies)
    new_indicies = (-score).argsort()
    new_indicies = old_indicies[new_indicies]
    return np.reshape(new_indicies, (1, new_indicies.size))

def prune_entity_types(types):
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

def get_entity_paths_in_ontology(entity_fine_types,leaf_to_root_paths_in_ontology):
    paths = {}
    entity_fine_types = prune_entity_types(entity_fine_types)
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

def _load_candidates(
    entity_catalogue, entity_encoding, faiss_index=None, index_path=None, logger=None
):
    candidate_encoding = entity_encoding
    # load all the 5903527 entities
    title2id = {}
    id2title = {}
    id2text = {}
    wikipedia_id2local_id = {}
    local_idx = 0
    with open(entity_catalogue, "r") as fin:
        lines = fin.readlines()
        for line in tqdm(lines):
            entity = json.loads(line)

            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()

                assert wikipedia_id not in wikipedia_id2local_id
                wikipedia_id2local_id[wikipedia_id] = local_idx

            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]
            local_idx += 1
    return (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
    )


if args.debug == False:

    (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id
    ) = _load_candidates(
            args.entity_catalogue,
            None,
            faiss_index=getattr(args, "faiss_index", None),
            index_path=getattr(args, "index_path", None),
            logger=logger,
        )
else:
    with open('zel_everything/pipeline_test/type_err_analysis/cache/title2id.json') as fp:
        title2id = json.load(fp)

    with open('zel_everything/pipeline_test/type_err_analysis/cache/id2title.json') as fp:
        id2title = json.load(fp)

    id2title = {int(k): v for k, v in id2title.items()}

    with open('zel_everything/pipeline_test/type_err_analysis/cache/wikipedia_id2local_id.json') as fp:
        wikipedia_id2local_id = json.load(fp)

    wikipedia_id2local_id = {int(k): v for k, v in wikipedia_id2local_id.items()}


    with open('zel_everything/pipeline_test/type_err_analysis/cache/id2text.json') as fp:
        id2text = json.load(fp)

    id2text = {int(k): v for k, v in id2text.items()}



id_to_type_dict = {}
id_to_fine_type_dict = {}

type_key = args.types_key
if args.debug == False:
    with open(args.entity_to_type_file_path, "r") as fin:
        lines = fin.readlines()
        for line in tqdm(lines):
            entity = json.loads(line)

            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()
            if type_key == "fine_types_id":
                id_to_fine_type_dict[wikipedia_id2local_id[wikipedia_id]] = entity["fine_types_id"]
                if args.ancestors:
                    id_to_type_dict[wikipedia_id2local_id[wikipedia_id]]= entity["fine_types_id"] + entity["coarse_types_id"]+entity["ancestors_types_id"]
                else:
                    id_to_type_dict[wikipedia_id2local_id[wikipedia_id]]= entity["fine_types_id"] + entity["coarse_types_id"]

            else:
                id_to_type_dict[wikipedia_id2local_id[wikipedia_id]]=entity[type_key]
    with open('zel_everything/pipeline_test/type_err_analysis/cache/cached_id_to_fine_type_dict.json', 'w') as fp:
        json.dump(id_to_fine_type_dict, fp)

    with open('zel_everything/pipeline_test/type_err_analysis/cache/cached_id_to_type_dict.json', 'w') as fp:
        json.dump(id_to_type_dict, fp)
else:
    with open(args.cached_id_to_type_dict_path) as f:
        id_to_type_dict = json.load(f)
    id_to_type_dict = {int(k): v for k, v in id_to_type_dict.items()}
    with open(args.cached_id_to_fine_type_dict_path) as f:
        id_to_fine_type_dict = json.load(f)
    id_to_fine_type_dict = {int(k): v for k, v in id_to_fine_type_dict.items()}


count_lines = 0
top_k_dict = {}
with jsonlines.open(args.topk_file_path) as f:
    for line in f.iter():
        #print(line)
        count_lines += 1
        line["id"]=int(line["id"].replace('_',''))
        top_predictions = [title2id[item] for item in line['top_10_predictions'] ]
        #top_k_dict[line["id"]]={'predictions': top_predictions,'rank_of_gold_entity':line['rank_of_gold_entity']}
        if line["id"] in top_k_dict:
            line["id"] = int(str(99) + str(line["id"]))

        top_k_dict[line["id"]]={'predictions': top_predictions,'scores':line['scores']}


print("No. of lines in top_k file {}".format(len(top_k_dict)))

id_to_type_probablity = {}
count_lines = 0
with jsonlines.open(args.type_probability_file_path) as f:
    for line in f.iter():
        #print(line)
        temp_dict = {}
        count_lines += 1
        id=int(line["id"])

        temp_dict['all_types_probability'] = json.loads(line['all_types_probability'])
        temp_dict['top_10_predicted_types'] = line['top_10_predicted_types']
        temp_dict['top_10_predicted_types_probability'] = line['top_10_predicted_types_probability']
        if id in id_to_type_probablity:
            id = int(str(99) + str(id))
            id_to_type_probablity[id] = temp_dict

            #print(id)
        else:
            id_to_type_probablity[id] = temp_dict


print(len(id_to_type_probablity))


with open(args.ontology_nodes_parent_file_path) as f:
    parent_of_node = json.load(f)

with open(args.type_co_occurrence_prob_file_path) as f:
  co_occurrence_matrix = json.load(f)
  co_occurrence_matrix = {int(k): v for k, v in co_occurrence_matrix.items()}

#print(parent_of_node)
leaf_nodes_ontology = get_leaf_nodes(parent_of_node)
#print(len(leaf_nodes_ontology))
#print(leaf_nodes_ontology)


leaf_to_root_paths_in_ontology = {}

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


#print("max_level in the ontology {}".format(max_level_in_ontology))
#print("max path in the ontology {}".format(max_path_in_ontology))

for k in [2,3,5,10,100]:
    labels = []
    nns_type = []
    nns_both = []
    nns_blink = []
    nns_multiply = []
    type_predictions_list = []


    instance_count = 0
    for id in tqdm(id_to_type_probablity):
        # print(id)
        # print(id_to_type_probablity[id])

        type_prediction = {}
        type_prediction['id'] = id
        sample = test_samples[id]
        label_id = wikipedia_id2local_id[int(sample['label_id'])]
        labels.extend([label_id])

        type_prediction['entity_name'] = id2title[label_id]
        type_prediction['gold_types'] =[id_to_type[str(item)] for item in id_to_type_dict[label_id]]

        type_prediction['top_10_predicted_types'] = id_to_type_probablity[id]['top_10_predicted_types']
        type_prediction['top_10_predicted_types_probability'] = id_to_type_probablity[id]['top_10_predicted_types_probability']
        #type_prediction['all_types_probability'] = id_to_type_probablity[id]['all_types_probability']
        type_prediction['context_left'] = sample['context_left']
        type_prediction['context_right'] = sample['context_right']
        type_prediction['mention'] = sample['mention']
        instance_count+=1
        # if instance_count > 5:
        #     break
        #print(test_samples[id])


        all_types_probablity  = id_to_type_probablity[id]['all_types_probability']
        path_mention_probablity = get_path_mention_probablity(leaf_to_root_paths_in_ontology,all_types_probablity)
        top_blink_predictions_for_mention = top_k_dict[id]['predictions']
        blink_scores = top_k_dict[id]['scores'][:k]
        type_based_scores = []
        for entity_id in top_blink_predictions_for_mention[:k]:
            # print(entity_id)
            #print(id2title[entity_id])
            #print(id_to_type_dict[entity_id])
            # print([id_to_type[str(item)] for item in id_to_type_dict[entity_id]])
            entity_fine_types= id_to_fine_type_dict[entity_id]
            entity_paths_in_ontology = get_entity_paths_in_ontology(entity_fine_types,leaf_to_root_paths_in_ontology)
            #print(len(entity_paths_in_ontology))

            type_score = 0
            for node in entity_paths_in_ontology:
                 new_entity_path = get_entity_probability(entity_paths_in_ontology[node],parent_of_node,co_occurrence_matrix)
                 type_score += new_entity_path['prob']*path_mention_probablity[node]['prob']
            #print(type_score)
            type_based_scores.append(type_score)
        type_based_scores = softmax(type_based_scores)
        blink_scores = softmax(blink_scores)

        total_scores_both = blink_scores + type_based_scores
        total_scores_type = type_based_scores
        total_scores_blink = blink_scores
        total_score_multiply = np.multiply(blink_scores, (1 + type_based_scores))

        new_indices_blink = calculate_indices(total_scores_blink, top_blink_predictions_for_mention)
        nns_blink.extend(new_indices_blink)

        new_indices_type = calculate_indices(total_scores_type, top_blink_predictions_for_mention)
        nns_type.extend(new_indices_type)

        new_indices_both = calculate_indices(total_scores_both, top_blink_predictions_for_mention)
        nns_both.extend(new_indices_both)

        new_indices_multiply = calculate_indices(total_score_multiply, top_blink_predictions_for_mention)
        nns_multiply.extend(new_indices_multiply)

        type_prediction['Prediction_Blink'] = [id2title[item] for item in new_indices_blink[0]]
        type_prediction['Prediction_Type'] = [id2title[item] for item in new_indices_type[0]]
        type_prediction['Prediction_Blink_Type'] = [id2title[item] for item in new_indices_both[0]]
        type_prediction['Prediction_Multiply'] = [id2title[item] for item in new_indices_multiply[0]]
        type_prediction['Blink_Score'] = blink_scores.tolist()
        type_prediction['Type_Score'] = type_based_scores.tolist()

        if new_indices_blink[0][0] == label_id and new_indices_type[0][0] == label_id:
            type_prediction['correct'] = '11'
        if new_indices_blink[0][0] == label_id and new_indices_type[0][0] != label_id:
            type_prediction['correct'] = '10'

        if new_indices_blink[0][0] != label_id and new_indices_type[0][0] == label_id:
            type_prediction['correct'] = '01'

        if new_indices_blink[0][0] != label_id and new_indices_type[0][0] != label_id:
            type_prediction['correct'] = '00'


        type_predictions_list.append(type_prediction)




    def calculate_scores(labels, nns_new, top_k ,score_type, NTS_count,logger):
        biencoder_accuracy = -1
        recall_at = -1
        # get recall values
        x = []
        y = []
        MRR = [0] * (top_k + 1)
        for i in range(1, top_k+1):
            temp_y = 0.0
            for label, top in zip(labels, nns_new):
                if label in top[:i]:
                    #print(label)
                    temp_y += 1
                    if i == 1:
                        MRR[i] += 1
                    elif label not in top[: (i - 1)]:
                        MRR[i] += 1

            if len(labels) > 0:
                temp_y /= len(labels)
            x.append(i)
            y.append(temp_y)
        # plt.plot(x, y)
        biencoder_accuracy = y[0]
        recall_at = y[-1]
        #print("Numbers for score type {}".format(score_type))
        logger.info("Scoring Method Used: {}".format(score_type))
        #print("biencoder accuracy: %.4f" % biencoder_accuracy)
        logger.info("biencoder accuracy: %.4f" % biencoder_accuracy)

        if len(y) > 4:
            #print("biencoder recall@%d: %.4f" % (5, y[4]))
            logger.info("biencoder recall@%d: %.4f" % (5, y[4]))

        #print("biencoder recall@%d: %.4f" % (top_k, y[-1]))
        logger.info("biencoder recall@%d: %.4f" % (top_k, y[-1]))

        for i in range(1, top_k):
            MRR[0] += MRR[i] / i
        if len(labels) > 0:
            #print("MRR: {}".format(MRR[0] / len(labels)))
            logger.info("MRR: {}".format(MRR[0] / len(labels)))

        #print("Normalized Test Set Size {}".format(NTS_count))
        logger.info("Normalized Test Set Size {}".format(NTS_count))

        # with open(os.path.join(args.output_path, 'top_k_wtth_prob.json'), 'w') as f:
        #     json.dump(top_k_dict, f)

        #print("after re-ranking normalized accuracy: {}".format(biencoder_accuracy * len(test_tensor_data) / NTS_count))
        #logger.info("after re-ranking normalized accuracy: {}".format(biencoder_accuracy * len(test_tensor_data) / NTS_count))

        #print("=============================")
        logger.info("=============================")
        return biencoder_accuracy


    logger.info("test file path: {}".format(args.data_path))
    logger.info("ancestors are used: {}".format(args.ancestors))
    logger.info("top_k considered: {}".format(k))
    logger.info("max level considered: {}".format(max_level))


    NTS_count = len(test_samples)
    calculate_scores(labels,nns_blink,k,"Blink Only", NTS_count,logger)
    calculate_scores(labels,nns_type,k,"Type Only", NTS_count,logger)
    calculate_scores(labels,nns_both,k,"Blink + Type", NTS_count,logger)
    calculate_scores(labels,nns_multiply,k,"Multiply Blink and Type Score", NTS_count,logger)


    def dump_jsonl(data, output_path, append=False):
        """
        Write list of objects to a JSON lines file.
        """
        mode = 'a+' if append else 'w'
        with open(output_path, mode, encoding='utf-8') as f:
            for line in data:
                json_record = json.dumps(line, ensure_ascii=False)
                f.write(json_record + '\n')
        print('Wrote {} records to {}'.format(len(data), output_path))

    output_prediction_file_path = os.path.join(args.output_path,"path_based_predictions_top_k_"+str(k)+"_level_"+str(max_level)+".jsonl")
    dump_jsonl(type_predictions_list,output_prediction_file_path)
