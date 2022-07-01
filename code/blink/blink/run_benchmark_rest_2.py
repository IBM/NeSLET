# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
from json import JSONDecoder

import blink.main_dense_new as main_dense
import argparse
import json
from json.encoder import JSONEncoder
import blink.candidate_ranking.utils as utils
from flask import Flask, request,jsonify
from wikimapper import WikiMapper
import pickle

app = Flask(__name__)



mapper = WikiMapper("NeSLET_everything/ZEL/data/wikimap_data/index_enwiki-latest.db")

models_path="NeSLET_everything/NeSLET/code/blink/blink/trained_models/hnm_5percent/"

config = {
        "test_entities": None,
        "test_mentions": None,
        "interactive": False,
        "top_k": 100,
        "biencoder_model": models_path + "pytorch_model.bin",
        "biencoder_config": models_path + "config.json",

        "entity_catalogue":   "NeSLET_everything/NeSLET/code/blink/blink/trained_models/hnm_5percent/entity.jsonl",

        "biencoder_training_params":models_path+"training_params.txt",


        "entity_encoding": models_path + "0_-1_old.t7",

        "crossencoder_model": models_path + "crossencoder_wiki_large.bin",
        "crossencoder_config": models_path + "crossencoder_wiki_large.json",
        "fast": True,  # set this to be true if speed is a concern
        "no_cuda": True,
        "output_path": "logs/"  # logging directory
        }


with open("NeSLET_everything/NeSLET/code/blink/blink/trained_models/hnm_5percent/wiki2dbr.pkl", "rb") as filein:
     wikimap = pickle.load(filein, encoding="utf-8")

entity_cat=config["entity_catalogue"]

new_dict = {}

title2id = {}
id2title = {}
id2text = {}
wikipedia_id2local_id = {}
local_idx = 0

with open(entity_cat, "r", encoding="utf-8") as fin:
    lines = fin.readlines()
    for line in lines:
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

id2url = {v: k for k, v in wikipedia_id2local_id.items()}


######################################################################################

args = argparse.Namespace(**config)

logger = utils.get_logger(args.output_path)

models = main_dense.load_models(args, logger=None,no_cuda=args.no_cuda)


def get_prediction(data, topk_user):
    (
        biencoder_accuracy,
        recall_at,
        crossencoder_normalized_accuracy,
        overall_unormalized_accuracy,
        num_datapoints,
        predictions,
        scores,
    ) = main_dense.run(args.no_cuda, args, logger, *models, test_data=data)
    s= [item[0:topk_user] for item in scores]
    p= [item[0:topk_user] for item in predictions]
    #return scores, predictions
    return s,p
###############################################################################################
### Below functions are written to retrieve wikidata id of corresponding  predicted wikipedia entity

###############################################################################################

def get_wikidata_id(list_title):
    id_list=[]
    for title in list_title:
        title_mod=title.replace(" ","_")
        wikidata_id = mapper.title_to_id(title_mod)
        wikidata_id_mod=str(wikidata_id)
        id_list.append(wikidata_id_mod)
    return id_list

def get_wiki_id_list_pred(predicted):
    predicted_wiki_data_list=[]
    for p in predicted:
        inner_list=get_wikidata_id(p)
        predicted_wiki_data_list.append(inner_list)
    return predicted_wiki_data_list
#############################################################################################
########## Below functions are written to retrieve dbpedia entity  of corresponding predicted wikipedia entity

#############################################################################################

def convert_wiki_dbpedia(list_title):
    id_list=[]
    for title in list_title:
        if title in title2id and title2id[title] in id2url and id2url[title2id[title]] in wikimap:
            e_url = wikimap[id2url[title2id[title]]]
            e_url = bytes(e_url, 'iso-8859-1').decode('utf-8')
            id_list.append(e_url)
        else:
            e_url="None"
            id_list.append(e_url)
    return id_list


def get_dbp_entity_list_pred(predicted):
    dbp_entity_list=[]
    for p in predicted:
        inner_list=convert_wiki_dbpedia(p)
        dbp_entity_list.append(inner_list)
    return dbp_entity_list



################################################################################################

################################################################################################

@app.route('/predict', methods=['GET'])
#@app.route('/multiple_predict', methods=['POST'])
def predict():
    if request.method == 'GET':
        data_to_link = []
        list_id=[]
        input_dict = {}
        if 'mention' in request.args and 'context_left' in request.args and 'context_right' in request.args:
            # if 'id' in request.args and 'label' in request.args and 'mention' in request.args and 'context_left' in request.args and 'context_right' in request.args:
            input_dict['id'] = int(request.args['id'])
            input_dict['label_id'] = -1
            input_dict['label'] = 'unknown'
            input_dict['mention'] = request.args['mention'].lower()
            input_dict['context_left'] = request.args['context_left'].lower()
            input_dict['context_right'] = request.args['context_right'].lower()
            list_id.append(input_dict["id"])
            data_to_link.append(input_dict)
            #t_k=int(request.args['topk_user'])
        else:
            #t_k=10
            return "Error: Either mention, left context or right context field is not provided. Please specify all of these."
        #if not any('topk_user' in d for d in input_data):
        if 'topk_user' in request.args:
            t_k=int(request.args['topk_user'])
        else:
            t_k=10

        s, pred = get_prediction(data_to_link, t_k)

        wiki_id_list = get_wiki_id_list_pred(pred)

        dbp_entity = get_dbp_entity_list_pred(pred)
        #res = {list_id[i]:pred[i] for i in range(len(list_id))}
        #res = {list_id[i]:str(list(zip(pred[i],s[i]))) for i in range(len(list_id))}
        res = {list_id[i]: str(list(zip(pred[i], wiki_id_list[i], s[i]))) for i in range(len(list_id))}
        #res = {list_id[i]: str(list(zip(pred[i], wiki_id_list[i], dbp_entity[i], s[i]))) for i in range(len(list_id))}
        return jsonify(res)


@app.route('/multiple_predict', methods=['POST'])

def multiple_predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        input_data=json.load(file)
        list_id=[]
        for i, line in enumerate(input_data):
            line["context_left"] = line["context_left"].lower()
            line["context_right"] = line["context_right"].lower()
            line["mention"] = line["mention"].lower()
            line["label"] = 'unknown'
            line["label_id"] = -1
            list_id.append(line["id"])

        if not any('topk_user' in d for d in input_data):
            t_k = 10
        else:
            t_k = next(d['topk_user'] for i, d in enumerate(input_data) if 'topk_user' in d)
        score, pred = get_prediction(input_data,t_k)
        #score, pred = get_prediction(input_data)
        wiki_id_list = get_wiki_id_list_pred(pred)
        dbp_entity = get_dbp_entity_list_pred(pred)
        
        dbp_entity = get_dbp_entity_list_pred(pred)
        #res = {list_id[i]: pred[i] for i in range(len(list_id))}
        #res = {list_id[i]: str(list(zip(pred[i],score[i]))) for i in range(len(list_id))}
        #res = {list_id[i]: str(list(zip(pred[i], wiki_id_list[i], score[i]))) for i in range(len(list_id))}
        res = {list_id[i]: str(list(zip(pred[i], wiki_id_list[i], dbp_entity[i], score[i]))) for i in range(len(list_id))}


        return jsonify(res)

@app.route('/json-example', methods=['POST'])

def json_example():
    list_of_id=[]
    request_data = request.get_json()
    for i, line in enumerate(request_data):
        line["context_left"] = line["context_left"].lower()
        line["context_right"] = line["context_right"].lower()
        line["mention"] = line["mention"].lower()
        line["label"] = 'unknown'
        line["label_id"] = -1
        list_of_id.append(line["id"])
    if not any('topk_user' in d for d in request_data):
        t_k=10
    else:
        t_k=next(d['topk_user'] for i,d in enumerate(request_data) if 'topk_user' in d)

    #score,pred = get_prediction(request_data)
    score, pred = get_prediction(request_data,t_k)
    wiki_id_list=get_wiki_id_list_pred(pred)

    dbp_entity=get_dbp_entity_list_pred(pred)
    #res = {list_of_id[i]:pred[i] for i in range(len(list_of_id))}
    #res = {list_of_id[i]: str(list(zip(pred[i], score[i]))) for i in range(len(list_of_id))}
    res = {list_of_id[i]: str(list(zip(pred[i],wiki_id_list[i], score[i]))) for i in range(len(list_of_id))}
    #res = {list_of_id[i]: str(list(zip(pred[i], wiki_id_list[i],dbp_entity[i], score[i]))) for i in range(len(list_of_id))}
    return jsonify(res)



if __name__ == '__main__':
    app.run('127.0.0.1')