import blink.main_dense as main_dense
import argparse
import json
import jsonlines
import blink.candidate_ranking.utils as utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--biencoder_model", type=str, help="Path to the biencoder model.")
    parser.add_argument("--biencoder_config", type=str, help="Path to the biencoder configuration.")
    parser.add_argument("--biencoder_training_params", type=str)
    parser.add_argument("--entity_encoding", type=str, help="Path to the entity catalogue.")
    #################################################
    parser.add_argument("--entity_catalogue", type=str,
                        default="zel_everything/scratch-shared/facebook_original_models/entity.jsonl",
                        help="Path to the entity catalogue.")
    # crossencoder
    parser.add_argument("--crossencoder_model", type=str,
                        default="zel_everything/scratch-shared/facebook_original_models/crossencoder_wiki_large.bin",
                        help="Path to the crossencoder model.")
    parser.add_argument("--crossencoder_config", type=str,
                        default="zel_everything/scratch-shared/facebook_original_models/crossencoder_wiki_large.json",
                        help="Path to the crossencoder configuration.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of candidates retrieved by biencoder.")
    parser.add_argument("--faiss_index", type=str, default=None, help="whether to use faiss index")
    parser.add_argument("--index_path", type=str, default=None, help="path to load indexer")
    parser.add_argument("--fast", dest="fast", default=True, action="store_true", help="only biencoder mode")
    parser.add_argument("--test_mentions", default=None)
    parser.add_argument("--test_entities", default=None)
    parser.add_argument("--interactive", "-i", default=False, action="store_true", help="Interactive mode.")
    #################################################
    parser.add_argument("--output_file_path", type=str, help="Path to the output.")
    parser.add_argument("--input_file_path", type=str)
    #parser.add_argument("--output_path", type=str, help="Path to the output.")

    args = parser.parse_args()
    input_file = args.input_file_path
    output_file = args.output_file_path
    logger = utils.get_logger(args.output_path)

    models = main_dense.load_models(args, logger)
    #models = main_dense.load_models(args, logger=None)

    # with open(input_file) as f:
    #     all_questions = json.load(f)
    # len(all_questions)

    data_to_link = []
    for i, line in enumerate(open(input_file, 'r', encoding='utf-8')):
        try:
            each_line = json.loads(line)
            english_dict = {'id': "", 'label': "","label_id":"", "context_left":"","mention":"","context_right":""}
            english_dict['id'] =  each_line["id"]
            english_dict['label'] =  each_line["label_title"]
            english_dict['label_id'] = each_line["label_id"]
            english_dict['context_left'] =  each_line['context_left'].lower()
            english_dict['context_right'] =  each_line['context_right'].lower()
            english_dict['mention'] =  each_line['mention'].lower()
            data_to_link.append(english_dict)
        except ValueError as err:
            print(err)
            print(each_line["id"])
            continue
    #predictions = main_dense.run(args, None, *models, test_data=data_to_link)

    (
        biencoder_accuracy,
        recall_at,
        crossencoder_normalized_accuracy,
        overall_unormalized_accuracy,
        num_datapoints,
        predictions,
        scores,
    ) = main_dense.run(args, logger, *models,test_data=data_to_link)
    #_, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)
    print("prediction completed")

    with open(output_file, 'w') as outfile:
        json.dump(predictions, outfile, indent=2)

    print("Done")
if __name__ == '__main__':
	main()

