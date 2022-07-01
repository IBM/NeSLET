import json
import jsonlines
import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument("--training_file_path",type=str)
parser.add_argument("--prediction_file_path",type=str)
parser.add_argument("--output_file_path",type=str)
args = parser.parse_args()

filename=args.training_file_path
prediction_file = args.prediction_file_path
output_file = args.output_file_path

output_dict={"id":"","top_10_predictions":[], "rank_of_gold_entity":"None"}

with open(prediction_file,'r') as fp:
     pred_list = json.load(fp)

print("prediction_length is",len(pred_list))
with jsonlines.open(output_file,mode='w') as writer:
    for i, line in enumerate(open(filename, 'r', encoding='utf-8')):
        #count=0
        try:
            each_line = json.loads(line)
            gold_entity= each_line['label_title']
            output_dict['id'] = each_line['id']
            output_dict['top_10_predictions']= pred_list[i]
            if gold_entity in pred_list[i]: #prediction:
                output_dict["rank_of_gold_entity"]=pred_list[i].index(gold_entity)
            else:
                output_dict["rank_of_gold_entity"] ="None" 
            output_dict.update(output_dict)
            writer.write(output_dict)
        except ValueError as err:
            print(err)
            print(line)
            continue

print('done')
