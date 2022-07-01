import argparse
import numpy as np
from scipy.special import softmax
from collections import defaultdict
import json
import jsonlines
from type_based_inference import perform_type_based_inference

parser = argparse.ArgumentParser()
parser.add_argument("--topk_file_path",type=str,help="Top-K prediction file path")
parser.add_argument("--type_probability_file_path",type=str,help="Type probability file path")

args = parser.parse_args()


count_lines = 0
top_k_dict = {}
with jsonlines.open(args.topk_file_path) as f:
    for line in f.iter():
        #print(line)
        count_lines += 1
        # top_predictions = [title2id[item] for item in line['top_10_predictions'] ]
        top_predictions = line['top_10_predictions']
        # #top_k_dict[line["id"]]={'predictions': top_predictions,'rank_of_gold_entity':line['rank_of_gold_entity']}
        top_k_dict[line["id"]]={'predictions': top_predictions,'scores':line['scores']}


print("No. of lines in top_k file {}".format(count_lines))

id_to_type_probablity = {}
count_lines = 0
with jsonlines.open(args.type_probability_file_path) as f:
    for line in f.iter():

        if count_lines in id_to_type_probablity:
            print("id all ready there")
        else:
            id_to_type_probablity[count_lines] = line
        count_lines += 1


print(len(id_to_type_probablity))



levels_to_prune = 3

our_results = []
for i in range(len(id_to_type_probablity)):

    # perform_type_based_inference function is function to perform type based inference
    # top_k_entities is a list of top_k entities returned by BLINK
    # top_k_entities_scores a list of scores of top_k entities returned by BLINK
    # type_probablities is a list of type probabilities
    # levels_to_prune is of type int and used to prune ontology tree at a level

    results_1 = perform_type_based_inference(top_k_dict[i]['predictions'],top_k_dict[i]['scores'],id_to_type_probablity[i],levels_to_prune)
    results_2 = perform_type_based_inference(top_k_dict[i]['predictions'],top_k_dict[i]['scores'],id_to_type_probablity[i])

    #print(results)
    # our_results.append(results_1['Prediction_Blink_Plus_Type'])
    # our_results.append(results_2['Prediction_Blink_Plus_Type'])
    our_results.append(results_1['Score_Type'])
    our_results.append(results_2['Score_Type'])
    #print(results_1['Score_Type'])
