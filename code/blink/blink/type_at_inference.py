import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
import json
import jsonlines
import torch
import argparse
from blink.biencoder.biencoder import TypedBiEncoderRanker,TypedBiEncoderRanker2
from blink.main_dense import _load_candidates,_get_test_samples,_process_biencoder_dataloader
import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
import io
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
import torch.nn as nn
import numpy as np
from scipy.special import softmax
from collections import defaultdict


parser = argparse.ArgumentParser()

parser.add_argument("--biencoder_model",type=str, help="Path to the biencoder model." )
parser.add_argument("--biencoder_config", type=str,help="Path to the biencoder configuration.")
parser.add_argument("--biencoder_training_params", type=str)
parser.add_argument("--entity_encoding",type=str,help="Path to the entity catalogue.")
parser.add_argument("--topk_file_path",type=str,help="Top-K prediction file path")
parser.add_argument("--data_path",type=str, help="The path to the test data.")
parser.add_argument("--type_dict_path",type=str, help="The path to the type dictionaries")
parser.add_argument("--entity_catalogue",dest="entity_catalogue",type=str,help="Path to the entity catalogue.")
parser.add_argument("--fast", dest="fast", default= False, action="store_true", help="only biencoder mode")
parser.add_argument("--output_path", type=str, help="Path to the output.")
parser.add_argument("--debug",action="store_true",help="Whether to run in debug mode with only 200 samples.")
parser.add_argument("--top_k", default=10, type=int)
parser.add_argument("--eval_batch_size",default=1,type=int, help="Total batch size for evaluation.")
parser.add_argument("--entity_to_type_file_path",type=str, help="The path to the entity to type file")
parser.add_argument("--softmax_on_type_score", type=bool, default=False, help="Whether to apply softmax on type scores")
parser.add_argument("--which_score",type=str, help="what scores to use")
parser.add_argument("--type_embeddings_path",type=str, help="type embedding path")
parser.add_argument("--cached_id_to_type_dict_path", type=str, help="Path to the cached id_to_type_dict for debug mode")
parser.add_argument("--normalize_type",action="store_true",help="Whether to normalize type scores")
parser.add_argument("--ancestors", type=bool, default=False, help="Whether to use ancestor types in case of DBpedia types")
parser.add_argument("--mode", type=str, help="val or test")
parser.add_argument("--best_alpha", type=str, help="Best value of alpha interpolation")
parser.add_argument("--best_alpha_fget", type=str, help="Best value of alpha for fget interpolation")

parser.add_argument("--probability_types",type=str,choices=["gold_top_k", "predicted"],default="predicted",)


args = parser.parse_args()
logger = utils.get_logger(args.output_path)

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


with open(args.type_dict_path) as f:
  type_dict = json.load(f)

count_lines = 0
top_k_dict = {}



def read_dataset(txt_file_path, debug=False):
    # file_name = "{}.jsonl".format(dataset_name)
    # txt_file_path = os.path.join(preprocessed_json_data_parent_folder, file_name)

    samples = []

    with io.open(txt_file_path, mode="r", encoding="utf-8") as file:
        for line in file:
            samples.append(json.loads(line.strip()))
            if debug and len(samples) > 200:
                break

    return samples

def load_typed_biencoder(params):
    # Init model
    if params['type_model'] == 1:
        biencoder = TypedBiEncoderRanker(params)
    if params['type_model'] == 2:
        biencoder = TypedBiEncoderRanker2(params)


    return biencoder

def load_models(args, logger=None):

    # load biencoder model
    if logger:
        logger.info("loading biencoder model")

    if args.biencoder_training_params is not None:
        tr_params = args.biencoder_training_params
        try:
            with open(tr_params, "r") as fp:
                biencoder_training_params = json.load(fp)
        except json.decoder.JSONDecodeError:
            with open(tr_params) as fp:
                for line in fp:
                    line = line.replace("'", '"')
                    line = line.replace("True", "true")
                    line = line.replace("False", "false")
                    line = line.replace("None", "null")
                    biencoder_training_params = json.loads(line)
                    break
    ####################################################
    if args.biencoder_training_params is not None:
        try:
            with open(args.biencoder_config) as json_file:
                biencoder_params = json.load(json_file)
        except json.decoder.JSONDecodeError:
            with open(args.biencoder_config) as json_file:
                for line in json_file:
                    line = line.replace("'", '"')
                    line = line.replace("True", "true")
                    line = line.replace("False", "false")
                    line = line.replace("None", "null")
                    biencoder_params = json.loads(line)
                    break
    else:
        with open(args.biencoder_config) as json_file:
            biencoder_params = json.load(json_file)
    ####################################################
    biencoder_params["path_to_model"] = args.biencoder_model
    biencoder_params["debug"] = False
    biencoder_params["no_cuda"] = False
    ####################################################################
    if args.biencoder_training_params is not None:
        # biencoder_params["entity_dict_path"] = args.entity_catalogue
        biencoder_params["bert_model"] = biencoder_training_params["bert_model"]
        biencoder_params["lowercase"] = biencoder_training_params["lowercase"]
        biencoder_params["out_dim"] = biencoder_training_params["out_dim"]
        biencoder_params['num_types'] = biencoder_training_params["num_types"]
        biencoder_params['max_type_list_len'] = biencoder_training_params["max_type_list_len"]
        biencoder_params['types_key'] = biencoder_training_params["types_key"]
        biencoder_params['context_key'] = biencoder_training_params["context_key"]
        if "type_embeddings_path" in biencoder_training_params:
            biencoder_params['type_embeddings_path'] = args.type_embeddings_path
        else:
            biencoder_params['type_embeddings_path'] = ""

        #biencoder_params['type_embeddings_path'] = ""


        if 'no_linear_after_type_embeddings' in biencoder_training_params:
            biencoder_params['no_linear_after_type_embeddings'] = biencoder_training_params["no_linear_after_type_embeddings"]
        else:
            biencoder_params['no_linear_after_type_embeddings'] = False

        if 'type_model' in biencoder_training_params:
            biencoder_params['type_model'] = biencoder_training_params["type_model"]
        else:
            biencoder_params['type_model'] = 1

        if 'freeze_type_embeddings' in biencoder_training_params:
            biencoder_params['freeze_type_embeddings'] = biencoder_training_params["freeze_type_embeddings"]
        else:
            biencoder_params['freeze_type_embeddings'] = False

        if 'freeze_context_bert' in biencoder_training_params:
            biencoder_params['freeze_context_bert'] = biencoder_training_params["freeze_context_bert"]
        else:
            biencoder_params['freeze_context_bert'] = False


        biencoder_params['type_embedding_dim'] = biencoder_training_params["type_embedding_dim"]
        biencoder_params['blink_loss_weight'] = biencoder_training_params["blink_loss_weight"]
        biencoder_params['type_loss_weight'] = biencoder_training_params["type_loss_weight"]
        biencoder_params['tb'] = biencoder_training_params["tb"]




        biencoder_params["pull_from_layer"] = biencoder_training_params[
            "pull_from_layer"
        ]
        biencoder_params["add_linear"] = biencoder_training_params["add_linear"]
        biencoder_params["silent"] = biencoder_training_params["silent"]

        biencoder_params["max_seq_length"] = biencoder_training_params["max_seq_length"]
        biencoder_params["max_context_length"] = biencoder_training_params[
            "max_context_length"
        ]
        biencoder_params["max_cand_length"] = biencoder_training_params[
            "max_cand_length"
        ]
    ####################################################################
    ############## DG added the following lines ##############
    biencoder_params["data_parallel"] = True
    biencoder_params[
        "eval_batch_size"
    ] = args.eval_batch_size  # 256 and 512 will throw CUDA out of memory error.

    biencoder = load_typed_biencoder(biencoder_params)



    # load candidate entities
    if logger:
        logger.info("loading candidate entities")
    (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
    ) = _load_candidates(
        args.entity_catalogue,
        args.entity_encoding,
        faiss_index=getattr(args, "faiss_index", None),
        index_path=getattr(args, "index_path", None),
        logger=logger,
    )

    return (
        biencoder,
        biencoder_params,
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
    )


(
    biencoder,
    biencoder_params,
    candidate_encoding,
    title2id,
    id2title,
    id2text,
    wikipedia_id2local_id,
    faiss_indexer,
) = load_models(args)

id_to_type_dict = {}
type_key = biencoder_params['types_key']
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
                if args.ancestors:
                    id_to_type_dict[wikipedia_id2local_id[wikipedia_id]]= entity["fine_types_id"] + entity["coarse_types_id"]+entity["ancestors_types_id"]
                else:
                    id_to_type_dict[wikipedia_id2local_id[wikipedia_id]]= entity["fine_types_id"] + entity["coarse_types_id"]

            else:
                id_to_type_dict[wikipedia_id2local_id[wikipedia_id]]=entity[type_key]
else:
    with open(args.cached_id_to_type_dict_path) as f:
        id_to_type_dict = json.load(f)
    id_to_type_dict = {int(k): v for k, v in id_to_type_dict.items()}

print(candidate_encoding.shape)
test_samples = _get_test_samples(args.data_path,None,title2id,wikipedia_id2local_id,logger)

count_lines = 0
top_k_dict = {}
with jsonlines.open(args.topk_file_path) as f:
    for line in f.iter():
        #print(line)
        count_lines += 1
        line["id"]=int(line["id"].replace('_',''))
        top_predictions = [title2id[item] for item in line['top_10_predictions'] ]
        #top_k_dict[line["id"]]={'predictions': top_predictions,'rank_of_gold_entity':line['rank_of_gold_entity']}
        top_k_dict[line["id"]]={'predictions': top_predictions,'scores':line['scores']}


print("No. of lines in top_k file {}".format(count_lines))


if args.ancestors == True:
    positive_types_strategy= "lflc_ancestor"
else:
    positive_types_strategy= "lflc"


test_data, test_tensor_data = data.process_mention_data_with_types_inference(
            samples=test_samples,
            tokenizer=biencoder.tokenizer,
            max_context_length=biencoder_params["max_context_length"],
            max_cand_length=biencoder_params["max_cand_length"],
            silent=biencoder_params["silent"],
            max_type_list_len=biencoder_params["max_type_list_len"],
            num_types=biencoder_params["num_types"],
            types_key=biencoder_params["types_key"],
            context_key=biencoder_params["context_key"],
            label_key="label",
            debug=args.debug,
            logger=logger,
            entities_file=args.entity_catalogue,
            positive_types_strategy=positive_types_strategy,
)
eval_batch_size = biencoder_params["eval_batch_size"]
test_sampler = SequentialSampler(test_tensor_data)
test_dataloader = DataLoader(
        test_tensor_data, sampler=test_sampler, batch_size=eval_batch_size
    )


print("test set size {}".format(len(test_tensor_data)))
print(args.softmax_on_type_score)

device = biencoder.device
n_gpu = biencoder.n_gpu

iter_ = tqdm(test_dataloader, desc="Batch")
biencoder.model.eval()
bce_loss_function = nn.BCEWithLogitsLoss(reduction="mean")

def calculate_indices(score,old_indicies):
    new_indicies = (-score).argsort()
    new_indicies = old_indicies[new_indicies]
    return np.reshape(new_indicies, (1, new_indicies.size))


labels = []
nns = []
all_scores = []
all_type_scores = []
blink_scores_normalized = []
all_sum_type_blink_scores = []
soft_max_fun = nn.Softmax(dim=1)
nns_new = []
nns_both = []
nns_type = []
nns_blink = []
nns_fget = []
nns_gold_type = []
nns_multiply = []
nns_multiply_both_types = []
nns_blink_fget = []
nns_interpolation = defaultdict(list)
nns_interpolation_fget = defaultdict(list)
nns_blink_original = []



NTS_count = 0
type_predictions_list = []
for step, batch in enumerate(iter_):

    batch = tuple(t.to(device) for t in batch)
    context_input, candidate_input, label_ids, type_label_vec,ids = batch


    if biencoder_params['type_model'] == 1:
        original_scores, type_scores = biencoder.forward_type_inference(
            context_input=context_input,
            cand_input=candidate_input,
            type_labels=type_label_vec,
            # cand_encs=candidate_encoding[top_k_dict[ids.item()]['predictions'],:].to(device)
            cand_encs=candidate_encoding

        )
    if biencoder_params['type_model'] == 2:
        original_scores, type_scores = biencoder.score_candidate(
            text_vecs=context_input,
            cand_vecs=candidate_input,
            random_negs=True,
            cand_encs=candidate_encoding

        )





    #print(type_scores.shape)
    probabilty_type_given_mention = torch.sigmoid(type_scores)
    probabilty_type_given_mention = probabilty_type_given_mention.cpu().data.numpy()
    complementary_probabilty_type_given_mention = 1-probabilty_type_given_mention

    #print(probabilty_type_given_mention.shape)
    #type_loss = bce_loss_function(type_scores, type_label_vec.to(torch.float))

    # type_labels = type_label_vec[0].cpu().detach().numpy()
    # probabilty_type_given_mention = probabilty_type_given_mention[0].cpu().detach().numpy()
    # type_based_score = 0.0
    # for index, item in enumerate(type_labels):
    #     if item == 1:
    #         type_based_score += probabilty_type_given_mention[index]
    #     else:
    #         type_based_score +=  (1 - probabilty_type_given_mention[index])


    original_scores, original_indicies = original_scores.topk(args.top_k)
    #original_blink_scores = soft_max_fun(original_scores)
    original_scores = original_scores.cpu().data.numpy()
    original_indicies = original_indicies.cpu().data.numpy()
    nns_blink_original.extend(original_indicies)


    ids=ids.cpu().tolist()
    temp_indices = [top_k_dict[x[0]]['predictions'][:args.top_k] for x in ids]
    indicies = np.array(temp_indices)
    temp_scores = [top_k_dict[x[0]]['scores'][:args.top_k] for x in ids]
    scores = np.array(temp_scores)
    blink_scores = softmax(scores)
    #blink_scores = scores

    probabilty_gold_types = np.zeros(shape=probabilty_type_given_mention.shape)
    probabilty_top_k_gold_types = np.zeros(shape=probabilty_type_given_mention.shape)


    label_ids = label_ids.cpu().data.numpy()
    for ii, gold_label in enumerate(label_ids):
        # print(gold_label[0])
        # print(ii)
        type_prediction = {}
        type_prediction['id'] = ids[ii][0]
        type_prediction['entity_name'] =  id2title[gold_label[0]]
        type_prediction['gold_types'] = [type_dict[str(item)] for item in id_to_type_dict[gold_label[0]]]
        gold_type_ids = [item for item in id_to_type_dict[gold_label[0]]]
        probabilty_gold_types[ii][gold_type_ids] = 1
        top_10_prediction = probabilty_type_given_mention[ii].argsort()[-10:][::-1]

        type_prediction['top_10_predicted_types'] =  [type_dict[str(item)] for item in top_10_prediction]
        type_prediction['top_10_predicted_types_probability'] = str(list(probabilty_type_given_mention[ii][top_10_prediction]))
        type_prediction['all_types_probability'] = str(list(probabilty_type_given_mention[ii]))
        type_predictions_list.append(type_prediction)

        top_prediction_ids = probabilty_type_given_mention[ii].argsort()[-len(gold_type_ids):][::-1]
        probabilty_top_k_gold_types[ii][top_prediction_ids] = 1

        if gold_label in indicies[ii]:
            NTS_count += 1
    complementary_probabilty_gold_types = 1 - probabilty_gold_types
    complementary_probabilty_top_k_gold_types = 1 - probabilty_top_k_gold_types

    if args.probability_types == 'gold_top_k':
        probabilty_type_given_mention = probabilty_top_k_gold_types
        complementary_probabilty_type_given_mention = complementary_probabilty_top_k_gold_types


    #blink_scores = blink_scores.cpu().data.numpy()
    labels.extend(label_ids)
    nns.extend(indicies)
    all_scores.extend(scores)
    blink_scores_normalized.extend(blink_scores)

    num_types = biencoder_params['num_types']
    for batch_id in range(len(scores)):
        type_based_scores = []
        fget_based_score = []
        gold_type_based_scores = []
        for index in indicies[batch_id]:
            forward = np.zeros(num_types)
            reverse = np.ones(num_types)
            type_list = id_to_type_dict[index]
            forward[type_list] = 1
            reverse[type_list]=0
            s1 = np.dot(forward, probabilty_type_given_mention[batch_id])
            s2 = np.dot(reverse, complementary_probabilty_type_given_mention[batch_id])

            s1_gold = np.dot(forward, probabilty_gold_types[batch_id])
            s2_gold = np.dot(reverse, complementary_probabilty_gold_types[batch_id])

            normalized_constant_positive = (forward == 1).sum()
            normalized_constant_negative = (reverse == 1).sum()

            if args.normalize_type == True:
                if normalized_constant_positive != 0 and normalized_constant_negative !=0:
                    s_total = (s1/normalized_constant_positive)+(s2/normalized_constant_negative)
                    s_total_gold = (s1_gold/normalized_constant_positive)+(s2_gold/normalized_constant_negative)
                    s1 = (s1 / normalized_constant_positive)
                if normalized_constant_positive == 0:
                    s_total = (s2 / normalized_constant_negative)
                    s_total_gold = (s2_gold/normalized_constant_negative)
                if normalized_constant_negative == 0:
                    s_total = (s1 / normalized_constant_positive)
                    s_total_gold = (s1_gold/normalized_constant_positive)
                    s1 = (s1 / normalized_constant_positive)

            else:
                s_total = (s1 + s2) / num_types
                s_total_gold = (s1_gold+s2_gold)/num_types

            fget_based_score.append(s1)
            type_based_scores.append(s_total)
            gold_type_based_scores.append(s_total_gold)

        type_based_scores = np.array([type_based_scores])
        fget_based_score = np.array([fget_based_score])
        gold_type_based_scores = np.array([gold_type_based_scores])


        if args.softmax_on_type_score == True:
            type_based_scores = softmax(type_based_scores)
            fget_based_score = softmax(fget_based_score)
            gold_type_based_scores = softmax(gold_type_based_scores)



        all_type_scores.extend(type_based_scores)

        # if args.which_score == 'both':
        #     total_score = blink_scores[batch_id] + type_based_scores
        # if args.which_score == 'type':
        #     total_score = type_based_scores
        # if args.which_score == 'blink':
        #     total_score = blink_scores[batch_id]


        # new_indices = (-total_score).argsort()
        # new_indices = indicies[batch_id][new_indices]
        # nns_new.extend(np.reshape(new_indices,(1, new_indices.size)))

        total_score_both = blink_scores[batch_id] + type_based_scores
        total_score_type = type_based_scores
        total_score_blink = blink_scores[batch_id]
        total_score_fget = fget_based_score
        total_score_blink_fget = blink_scores[batch_id] + fget_based_score
        total_score_gold_type = gold_type_based_scores
        total_score_multiply = np.multiply(blink_scores[batch_id], (1+fget_based_score)[0])
        total_score_multiply_both_types = np.multiply(blink_scores[batch_id], (1+type_based_scores)[0])




        new_indices_both = (-total_score_both).argsort()
        new_indices_both = indicies[batch_id][new_indices_both]
        nns_both.extend(np.reshape(new_indices_both,(1, new_indices_both.size)))

        new_indices_type = (-total_score_type).argsort()
        new_indices_type = indicies[batch_id][new_indices_type]
        nns_type.extend(np.reshape(new_indices_type,(1, new_indices_type.size)))

        new_indices_blink = (-total_score_blink).argsort()
        new_indices_blink = indicies[batch_id][new_indices_blink]
        nns_blink.extend(np.reshape(new_indices_blink,(1, new_indices_blink.size)))

        new_indices_fget = (-total_score_fget).argsort()
        new_indices_fget = indicies[batch_id][new_indices_fget]
        nns_fget.extend(np.reshape(new_indices_fget,(1, new_indices_fget.size)))

        new_indices_blink_fget = (-total_score_blink_fget).argsort()
        new_indices_blink_fget = indicies[batch_id][new_indices_blink_fget]
        nns_blink_fget.extend(np.reshape(new_indices_blink_fget,(1, new_indices_blink_fget.size)))

        new_indices_gold_type = (-total_score_gold_type).argsort()
        new_indices_gold_type = indicies[batch_id][new_indices_gold_type]
        nns_gold_type.extend(np.reshape(new_indices_gold_type, (1, new_indices_gold_type.size)))


        new_indices_multiply = (-total_score_multiply).argsort()
        new_indices_multiply = indicies[batch_id][new_indices_multiply]
        nns_multiply.extend(np.reshape(new_indices_multiply, (1, new_indices_multiply.size)))

        new_indices_multiply_both_types = (-total_score_multiply_both_types).argsort()
        new_indices_multiply_both_types = indicies[batch_id][new_indices_multiply_both_types]
        nns_multiply_both_types.extend(np.reshape(new_indices_multiply_both_types, (1, new_indices_multiply_both_types.size)))

        if args.mode == 'val':
            for alpha in np.linspace(0.0,1.0,11):
                total_interpolation_score = blink_scores[batch_id] + alpha*type_based_scores
                new_indices_interpolation = (-total_interpolation_score).argsort()
                new_indices_interpolation = indicies[batch_id][new_indices_interpolation]
                nns_interpolation[alpha].extend(np.reshape(new_indices_interpolation, (1, new_indices_interpolation.size)))

                total_interpolation_fget_score = blink_scores[batch_id] + alpha * total_score_fget
                new_indices_interpolation_fget = (-total_interpolation_fget_score).argsort()
                new_indices_interpolation_fget = indicies[batch_id][new_indices_interpolation_fget]
                nns_interpolation_fget[alpha].extend(
                    np.reshape(new_indices_interpolation_fget, (1, new_indices_interpolation_fget.size)))


        else:
            total_interpolation_score = blink_scores[batch_id] + float(args.best_alpha) * type_based_scores
            new_indices_interpolation = (-total_interpolation_score).argsort()
            new_indices_interpolation = indicies[batch_id][new_indices_interpolation]
            nns_interpolation[float(args.best_alpha)].extend(np.reshape(new_indices_interpolation, (1, new_indices_interpolation.size)))

            total_interpolation_fget_score = blink_scores[batch_id] + float(args.best_alpha_fget) * total_score_fget
            new_indices_interpolation_fget = (-total_interpolation_fget_score).argsort()
            new_indices_interpolation_fget = indicies[batch_id][new_indices_interpolation_fget]
            nns_interpolation_fget[float(args.best_alpha_fget)].extend(
                np.reshape(new_indices_interpolation_fget, (1, new_indices_interpolation_fget.size)))

        #top_k_dict[ids[batch_id][0]]["type_prob"]=probabilty_type_given_mention.tolist()





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
    logger.info("after re-ranking normalized accuracy: {}".format(biencoder_accuracy * len(test_tensor_data) / NTS_count))

    #print("=============================")
    logger.info("=============================")
    return biencoder_accuracy

logger.info("results are generated using probablity type: {}".format(args.probability_types))
logger.info("path to model: {}".format(args.biencoder_model))
logger.info("path to config: {}".format(args.biencoder_config))
logger.info("path to training params file: {}".format(args.biencoder_training_params))
logger.info("top_k considered: {}".format(args.top_k))
logger.info("test file path: {}".format(args.data_path))
logger.info("ancestors are used: {}".format(args.ancestors))
logger.info("normalize_type is : {}".format(args.normalize_type))



calculate_scores(labels,nns_blink_original,args.top_k,"Original_Blink Score", NTS_count,logger)


calculate_scores(labels,nns_gold_type,args.top_k,"GOLD TYPE", NTS_count,logger)
calculate_scores(labels,nns_both,args.top_k,"Both", NTS_count,logger)
calculate_scores(labels,nns_type,args.top_k,"Type only", NTS_count,logger)
calculate_scores(labels,nns_multiply_both_types,args.top_k,"Multiply Blink and Both Type Score", NTS_count,logger)
calculate_scores(labels,nns_blink_fget,args.top_k,"Blink +  FGET", NTS_count,logger)
calculate_scores(labels,nns_multiply,args.top_k,"Multiply Blink and Type Score", NTS_count,logger)

calculate_scores(labels,nns_fget,args.top_k,"FGET", NTS_count,logger)
calculate_scores(labels,nns_blink,args.top_k,"Blink Only", NTS_count,logger)


if args.mode == 'val':
    interpolation_accuracy = 0.0
    best_alpha = 0.0
    for alpha in nns_interpolation:
        temp_accuracy = calculate_scores(labels, nns_interpolation[alpha], args.top_k, "interpolation_alpha_"+str(alpha), NTS_count, logger)
        if interpolation_accuracy < temp_accuracy:
            interpolation_accuracy = temp_accuracy
            best_alpha = alpha

    logger.info("*************************")

    logger.info("top_k: {} Best alpha: {} and Best interpolation accuracy: {}".format(args.top_k,best_alpha,interpolation_accuracy))

    logger.info("*************************")

    interpolation_accuracy_fget = 0.0
    best_alpha_fget = 0.0
    for alpha_fget in nns_interpolation_fget:
        temp_accuracy_fget = calculate_scores(labels, nns_interpolation_fget[alpha_fget], args.top_k,
                                         "interpolation_fget_alpha_" + str(alpha_fget), NTS_count, logger)
        if interpolation_accuracy_fget < temp_accuracy_fget:
            interpolation_accuracy_fget = temp_accuracy_fget
            best_alpha_fget = alpha_fget

    logger.info("*************************")
    logger.info("top_k: {} Best alpha_fget: {} and Best interpolation_fget accuracy: {}".format(args.top_k,best_alpha_fget, interpolation_accuracy_fget))
    logger.info("*************************")


else:
    best_alpha = float(args.best_alpha)
    interpolation_accuracy = calculate_scores(labels, nns_interpolation[best_alpha], args.top_k, "interpolation_alpha_" + str(best_alpha), NTS_count,
                     logger)
    #logger.info("*************************")
    logger.info("top_k: {} Best alpha: {} and Best interpolation accuracy: {}".format(args.top_k,best_alpha,interpolation_accuracy))
    logger.info("=============================")


    best_alpha_fget = float(args.best_alpha_fget)
    interpolation_accuracy_fget = calculate_scores(labels, nns_interpolation_fget[best_alpha_fget], args.top_k, "interpolation_fget_alpha_" + str(best_alpha_fget),
                     NTS_count,
                     logger)
    #logger.info("*************************")
    logger.info("top_k: {} Best alpha_fget: {} and Best interpolation_fget accuracy: {}".format(args.top_k,best_alpha_fget, interpolation_accuracy_fget))
    logger.info("=============================")



#dump_jsonl(type_predictions_list,args.output_type_prediction_path)


def add_predictions(test_samples, labels, nns_gold_type, nns_both, nns_type, nns_blink_fget,nns_multiply,nns_fget, nns_blink):

    index = 0
    for label, top_gold,top_both,top_type,top_blink_fget,top_multiply,top_fget,top_blink,top_interpolation,top_interpolation_fget in zip(labels, nns_gold_type, nns_both, nns_type , nns_blink_fget, nns_multiply,nns_fget, nns_blink,nns_interpolation[best_alpha],nns_interpolation_fget[best_alpha_fget]):
        id = test_samples[index]['id']
        assert type_predictions_list[index]['id'] == id
        type_predictions_list[index]['Prediction_Gold_Type'] = [id2title[item] for item in top_gold]
        type_predictions_list[index]['Prediction_Blink_Type'] = [id2title[item] for item in top_both]
        type_predictions_list[index]['Prediction_Type'] = [id2title[item] for item in top_type]
        type_predictions_list[index]['Prediction_Blink_PosType'] = [id2title[item] for item in top_blink_fget]
        type_predictions_list[index]['Prediction_Multiply'] = [id2title[item] for item in top_multiply]
        type_predictions_list[index]['Prediction_PosType'] = [id2title[item] for item in top_fget]
        type_predictions_list[index]['Prediction_Blink'] = [id2title[item] for item in top_blink]
        type_predictions_list[index]['Prediction_Interpolation'] = [id2title[item] for item in top_interpolation]
        type_predictions_list[index]['Prediction_Interpolation_fget'] = [id2title[item] for item in top_interpolation_fget]

        index += 1

#if args.mode == 'test':
add_predictions(test_samples,labels,nns_gold_type, nns_both, nns_type, nns_blink_fget,nns_multiply,nns_fget, nns_blink)
output_prediction_file_path = os.path.join(args.output_path,"predictions_top_k_"+str(args.top_k)+".jsonl")
dump_jsonl(type_predictions_list,output_prediction_file_path)
