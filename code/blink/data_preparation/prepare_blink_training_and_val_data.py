import json
import argparse
import os

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


def make_id_to_info_dict(entity_file_contents):
    id_to_info = {}
    for entity_item in entity_file_contents:
        curid = entity_item['idx'].split('curid=')[-1]
        id_to_info[curid] = entity_item
    return id_to_info


def write_jsonl(data_list, file_path):
    data_list_json = [json.dumps(d)+"\n" for d in data_list]
    with open(file_path, 'w') as out_file:
        out_file.writelines(data_list_json)
        
def process(input_file, output_file, id_to_info, num_examples_to_keep=-1):
    data_original = read_first_k_lines(input_file, num_examples_to_keep)
    
    new_data = []
    missing_curids = []

    for data_item in data_original:
        new_data_item = {}
        new_data_item["mention"] = data_item["meta"]["mention"]
        new_data_item["context_left"] = data_item["meta"]["left_context"]
        new_data_item["context_right"] = data_item["meta"]["right_context"]
        new_data_item["label_id"] = data_item["output"][0]["provenance"][0]["wikipedia_id"]
        new_data_item["label_title"] = data_item["output"][0]["provenance"][0]["title"]

        if new_data_item["label_id"] not in id_to_info:
            missing_curids.append(new_data_item["label_id"])
            continue

        new_data_item["label"] = id_to_info[new_data_item["label_id"]]["text"]
        new_data.append(new_data_item)
    
    print("Output data length: ", len(new_data))
    print("Number of missing curids: ",len(missing_curids))
    
    write_jsonl(data_list=new_data, file_path=output_file)
    print("Finished processing ", input_file)


if __name__ == "__main__":
    
#     usage: python prepare_blink_training_and_val_data.py --input_dir /blink_training_data/ --output_dir /blink_training_data_processed --num_examples_to_keep 16
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="The input directory should have entity.jsonl, train.jsonl and valid.jsonl")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_examples_to_keep", type=int, default=-1, help="-1 keeps all examples")
    config = parser.parse_args()
    
    entity_file_path = os.path.join(config.input_dir, "entity.jsonl")
    entity_file_contents = read_first_k_lines(entity_file_path, k=-1)
    id_to_info = make_id_to_info_dict(entity_file_contents)
    
    process(input_file=os.path.join(config.input_dir, "train.jsonl"), 
            output_file=os.path.join(config.output_dir, "train.jsonl"), 
            id_to_info=id_to_info, 
            num_examples_to_keep=config.num_examples_to_keep
           )
    
    process(input_file=os.path.join(config.input_dir, "valid.jsonl"), 
            output_file=os.path.join(config.output_dir, "valid.jsonl"), 
            id_to_info=id_to_info, 
            num_examples_to_keep=config.num_examples_to_keep
           )
    
    print("All done.")
    
    
    