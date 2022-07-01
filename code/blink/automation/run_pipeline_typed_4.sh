#!/bin/bash -x

#SBATCH -J test_job     # overridden by commandline argument              
#SBATCH -o NeSLET_everything/scratch/job_logs/stdout_%j # overridden by commandline argument
#SBATCH -e NeSLET_everything/scratch/job_logs/stderr_%j # overridden by commandline argument
#SBATCH --qos=dcs-48hr
#SBATCH --time=47:00:00
#SBATCH --gres=gpu:6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1


#---------------------------------------------------------------
# Example usage: 

# sbatch -J job -o NeSLET_everything/scratch/log/stdout_%j -e NeSLET_everything/scratch/log/stderr_%j --export=ALL,code_dir='NeSLET_everything/NeSLET/code/blink/blink',experiment_dir='NeSLET_everything/scratch/pipeline_out',training_dataset_name='pipeline_test',percentage_training_data_to_use=100,num_vanilla_training_epochs=2,num_hnm_training_epochs=2,types_key='fine_types_id',base_model='NeSLET_everything/scratch-shared/blink_output/p_blink_0_01/hnm_training_output_dir/pytorch_model.bin' pipeline_typed_4.sh

#---------------------------------------------------------------


# Pick these variables from the environment or uncomment and initialize them below

# # something like NeSLET_everything/NeSLET/code/blink/blink
# code_dir=NeSLET_everything/NeSLET/code/blink/blink

# # place to store all the outputs
# experiment_dir=NeSLET_everything/scratch/pipeline_output_dir

# # Allowed training_dataset_name : blink_wiki, fget_wiki_um, fget_wiki_conll, pipeline_test
# training_dataset_name="pipeline_test"

# # some number between 0 to 100
# percentage_training_data_to_use=50

# num_vanilla_training_epochs=1
# num_hnm_training_epochs=1

# "fine_types_id" or "fgetc_category_id"
# types_key="fgetc_category_id"


# base_model='NeSLET_everything/scratch-shared/blink_output/p_blink_0_01/hnm_training_output_dir/pytorch_model.bin'



if [[ $types_key =  "fine_types_id" ]]
then
   num_types=768
   type_embedding_dim=768
   type_loss_weight=1.0
   type_vectors_file=NeSLET_everything/scratch-shared/data/type_vectors/bert-base/dbpedia_2020_types.t7
elif [[ $types_key =  "fgetc_category_id" ]]
then
   num_types=60000
   type_embedding_dim=768
   type_loss_weight=1.0
   
   if [[ $training_dataset_name =  "fget_wiki_um" ]]
   then
      type_vectors_file=NeSLET_everything/scratch-shared/data/type_vectors/bert-base/unseen_60K_categories.t7
   elif [[ $training_dataset_name =  "fget_wiki_conll" ]]
   then
      type_vectors_file=NeSLET_everything/scratch-shared/data/type_vectors/bert-base/conll_60K_categories.t7
   elif [[ $training_dataset_name =  "pipeline_test" ]]
   then
      type_vectors_file=NeSLET_everything/scratch-shared/data/type_vectors/bert-base/dbpedia_2020_types.t7
   else
      echo "Type vectors file is unavailable. Pipeline aborted."
      exit
   fi

else
   echo "The given value of types_key is not supported. Pipeline aborted."
   exit
fi



#------------------

entities_file=NeSLET_everything/scratch-shared/facebook_original_models/entity.jsonl
saved_cand_ids_file=NeSLET_everything/scratch-shared/entity_token_ids_128.t7

#------------------

wiki_blink_training_file=NeSLET_everything/scratch-shared/saswati/data/100percent_dbpedia_type_desc_ance/train_dbpedia_types_desc_ances.jsonl
wiki_blink_val_file=NeSLET_everything/scratch-shared/saswati/data/100percent_dbpedia_type_desc_ance/valid_dbpedia_types_desc_ances.jsonl

## for um val and test
wiki_blink_val_2_file=NeSLET_everything/scratch-shared/data/fget_processed_data/unseen_mention_fget_data_processed/unseen_60K_mention_dev_10K_dbpedia_ancs_desc.jsonl
wiki_blink_test_file=NeSLET_everything/scratch-shared/data/fget_processed_data/unseen_mention_fget_data_processed/unseen_60K_mention_test_10K_updated_dbpedia_ancs_desc.jsonl

## for conll val and test
#wiki_blink_val_2_file=NeSLET_everything/scratch-shared/data/fget_processed_data/conll_fget_data_processed/conll_60K_mention_dev_4791_dbpedia_types_desc_ances.jsonl
#wiki_blink_test_file=NeSLET_everything/scratch-shared/data/fget_processed_data/conll_fget_data_processed/conll_60K_mention_test_4485_updated_dbpedia_types_desc_ances.jsonl


#------------------

wiki_fget_um_training_file=NeSLET_everything/scratch-shared/data/fget_processed_data/wiki_fget_data_processed/wiki_fget_mention_train_5.6M_for_unseen_60K_updated_dbpedia_types_desc_ances.jsonl
wiki_fget_um_val_file=NeSLET_everything/scratch-shared/data/fget_processed_data/wiki_fget_data_processed/new_um_valid_dbpedia_type_desc_ance.jsonl
wiki_fget_um_val_2_file=zel_everything/scratch-shared/data/fget_processed_data/unseen_mention_fget_data_processed/unseen_60K_mention_dev_10K_dbpedia_ancs_desc.jsonl
wiki_fget_um_test_file=zel_everything/scratch-shared/data/fget_processed_data/unseen_mention_fget_data_processed/unseen_60K_mention_test_10K_updated_dbpedia_ancs_desc.jsonl


#------------------

wiki_fget_conll_training_file=zel_everything/scratch-shared/data/fget_processed_data/wiki_fget_data_processed/wiki_fget_mention_train_6M_for_conll_60K_updated_dbpedia_types_desc_ances.jsonl
wiki_fget_conll_val_file=zel_everything/scratch-shared/data/fget_processed_data/conll_fget_data_processed/conll_60K_mention_dev_5K_dbpedia_types_desc_ances.jsonl
wiki_fget_conll_val_2_file=zel_everything/scratch-shared/data/fget_processed_data/conll_fget_data_processed/conll_60K_mention_dev_4791_dbpedia_types_desc_ances.jsonl
wiki_fget_conll_test_file=zel_everything/scratch-shared/data/fget_processed_data/conll_fget_data_processed/conll_60K_mention_test_4485_updated_dbpedia_types_desc_ances.jsonl


#------------------


if [[ $training_dataset_name =  "blink_wiki" ]]
then
   echo "Val 1 and val 2 paths are the same"
   training_data_file=$wiki_blink_training_file
   validation_data_file=$wiki_blink_val_file
   validation_2_data_file=$wiki_blink_val_2_file
   test_data_file=$wiki_blink_test_file
elif [[ $training_dataset_name =  "fget_wiki_um" ]]
then
   training_data_file=$wiki_fget_um_training_file
   validation_data_file=$wiki_fget_um_val_file
   validation_2_data_file=$wiki_fget_um_val_2_file
   test_data_file=$wiki_fget_um_test_file
elif [[ $training_dataset_name =  "fget_wiki_conll" ]]
then
   echo "Val 1 and val 2 paths are the same"
   training_data_file=$wiki_fget_conll_training_file
   validation_data_file=$wiki_fget_conll_val_file
   validation_2_data_file=$wiki_fget_conll_val_2_file
   test_data_file=$wiki_fget_conll_test_file
elif [[ $training_dataset_name =  "pipeline_test" ]]
then
   training_data_file=zel_everything/scratch-shared/data/small_data/train.jsonl
   validation_data_file=zel_everything/scratch-shared/data/small_data/valid.jsonl
   validation_2_data_file=zel_everything/scratch-shared/data/small_data/valid_2.jsonl
   test_data_file=zel_everything/scratch-shared/data/small_data/test.jsonl
   entities_file=zel_everything/scratch-shared/dineshk/test/entity_small.jsonl
   saved_cand_ids_file=zel_everything/scratch-shared/data/small_data/entity_token_ids_128.t7
else
   echo "The given value of training_dataset_name is not supported. Pipeline aborted."
   exit
fi


# create the directories inside $experiment_dir

mkdir $experiment_dir
vanilla_training_data_dir=${experiment_dir}/vanilla_training_data_dir ; mkdir $vanilla_training_data_dir
vanilla_training_output_dir=${experiment_dir}/vanilla_training_output_dir ; mkdir $vanilla_training_output_dir
vanilla_training_t7_dir=${experiment_dir}/vanilla_training_t7_dir ; mkdir $vanilla_training_t7_dir
vanilla_training_prediction_dir=${experiment_dir}/vanilla_training_prediction_dir ; mkdir $vanilla_training_prediction_dir 
hnm_training_data_dir=${experiment_dir}/hnm_training_data_dir ; mkdir $hnm_training_data_dir
hnm_training_output_dir=${experiment_dir}/hnm_training_output_dir ; mkdir $hnm_training_output_dir
hnm_training_t7_dir=${experiment_dir}/hnm_training_t7_dir ; mkdir $hnm_training_t7_dir
hnm_training_valid_2_prediction_dir=${experiment_dir}/hnm_training_valid_2_prediction_dir ; mkdir $hnm_training_valid_2_prediction_dir 
hnm_training_prediction_dir=${experiment_dir}/hnm_training_prediction_dir ; mkdir $hnm_training_prediction_dir 


pipeline_log_file=${experiment_dir}/pipeline_log.txt

echo "python interpreter: `which python`" >> $pipeline_log_file

echo "===== Data =====" >> $pipeline_log_file
echo "Training data source: $training_data_file" >> $pipeline_log_file
echo "Val data: $validation_data_file" >> $pipeline_log_file
echo "Test data: $test_data_file" >> $pipeline_log_file
echo "================" >> $pipeline_log_file

# 1. take $percentage_training_data_to_use of the $training_data_file, call it training.jsonl and copy it to $vanilla_training_data_dir
# 2. copy validation_data_file to $vanilla_training_data_dir

full_len=`wc -l $training_data_file | cut -d " " -f 1`
lines_to_keep=`python -c "print(int( ($percentage_training_data_to_use * $full_len)/100 ))"`

head -n $lines_to_keep $training_data_file > $vanilla_training_data_dir/train.jsonl
cp $validation_data_file $vanilla_training_data_dir/valid.jsonl


echo "Created training and val sets. Took ${lines_to_keep} from ${training_data_file}" >> $pipeline_log_file


# Vanilla training

echo "Starting vanilla training"  >> $pipeline_log_file

cd $code_dir ; PYTHONPATH=. python blink/biencoder/train_biencoder_with_types.py --data_path $vanilla_training_data_dir --output_path $vanilla_training_output_dir --learning_rate 1e-5 --train_batch_size 128 --eval_batch_size 512 --num_train_epochs $num_vanilla_training_epochs --data_parallel --bert_model bert-base-uncased --max_context_length 64 --eval_interval 9999999 --save_interval 1000 --zeshel False --lowercase --print_interval 100 --shuffle True --blink_loss_weight 1.0 --type_loss_weight $type_loss_weight --type_embedding_dim $type_embedding_dim --type_embeddings_path $type_vectors_file --no_linear_after_type_embeddings --type_model 4 --main_metric entity --num_types $num_types --types_key $types_key --path_to_model $base_model --type_network_learning_rate 0.001 --type_task_importance_scheduling loss_weight --cls_start 5 --cls_end 12


if [ $? -eq 0 ]; then
   echo "Vanilla training done"  >> $pipeline_log_file
else
   echo "Vanilla training has failed"  >> $pipeline_log_file
   exit
fi


# T7 generation

echo "Starting T7 generation"  >> $pipeline_log_file

cd $code_dir ;
python generate_candidates.py --path_to_model_config $vanilla_training_output_dir/config.json --path_to_model $vanilla_training_output_dir/pytorch_model.bin --entity_dict_path $entities_file --encoding_save_file_dir $vanilla_training_t7_dir --file_training_params $vanilla_training_output_dir/training_params.txt --saved_cand_ids $saved_cand_ids_file



if [ $? -eq 0 ]; then
   echo "T7 generation done"  >> $pipeline_log_file
else
   echo "T7 generation has failed"  >> $pipeline_log_file
   exit
fi


# Generate prediction of training data by running inference code

echo "Generating predictions on training set"  >> $pipeline_log_file

cd $code_dir;

python run_benchmark_command_line_pipeline.py --biencoder_config $vanilla_training_output_dir/config.json --biencoder_model $vanilla_training_output_dir/pytorch_model.bin --entity_encoding $vanilla_training_t7_dir/0_-1.t7 --biencoder_training_params $vanilla_training_output_dir/training_params.txt --output_path $vanilla_training_prediction_dir --dataset_name training_set --test_file_path $vanilla_training_data_dir/train.jsonl --output_file_path $vanilla_training_prediction_dir/prediction.json --fast --top_k 10 --output_score_file_path $vanilla_training_prediction_dir/scores.json


if [ $? -eq 0 ]; then
   echo "prediction generation on training set is done"  >> $pipeline_log_file
else
   echo "prediction generation on training set has failed"  >> $pipeline_log_file
   exit
fi


# Generating input for hnm model

echo "Preparing data for hnm training"  >> $pipeline_log_file

cd $code_dir;
python generate_biencoder_formatted_output.py --training_file_path $vanilla_training_data_dir/train.jsonl --prediction_file_path $vanilla_training_prediction_dir/prediction.json --output_file_path $hnm_training_data_dir/topk.jsonl 


if [ $? -eq 0 ]; then
   echo "hnm training data preparation is done"  >> $pipeline_log_file
else
   echo "hnm training data preparation has failed"  >> $pipeline_log_file
   exit
fi


# Do HNM training

echo "Starting HNM training"  >> $pipeline_log_file

cd $code_dir; 

PYTHONPATH=. python blink/biencoder/train_biencoder_with_types.py --data_path $vanilla_training_data_dir --output_path $hnm_training_output_dir --learning_rate 1e-5 --train_batch_size 66 --eval_batch_size 128 --num_train_epochs $num_hnm_training_epochs --data_parallel --bert_model bert-base-uncased --max_context_length 64 --eval_interval 9999999 --save_interval 1000 --zeshel False --lowercase --print_interval 100 --shuffle True --entities_file $entities_file --hard_negatives_file $hnm_training_data_dir/topk.jsonl --max_num_negatives 9 --path_to_model $vanilla_training_output_dir/pytorch_model.bin --blink_loss_weight 1.0 --type_loss_weight $type_loss_weight --type_embedding_dim $type_embedding_dim --type_embeddings_path $type_vectors_file --no_linear_after_type_embeddings --type_model 4 --main_metric entity --num_types $num_types --types_key $types_key --type_network_learning_rate 0.001 --type_task_importance_scheduling loss_weight --cls_start 5 --cls_end 12 --eval_set_paths $wiki_fget_um_val_2_file --eval_set_paths $wiki_fget_um_test_file --eval_set_paths $wiki_fget_conll_val_2_file --eval_set_paths $wiki_fget_conll_test_file


if [ $? -eq 0 ]; then
   echo "hnm training is done"  >> $pipeline_log_file
else
   echo "hnm training has failed"  >> $pipeline_log_file
   exit
fi


# Generate t7 file for hnm model

echo "Generating T7"  >> $pipeline_log_file

cd $code_dir;
python generate_candidates.py --path_to_model_config $hnm_training_output_dir/config.json --path_to_model $hnm_training_output_dir/pytorch_model.bin --entity_dict_path $entities_file --encoding_save_file_dir $hnm_training_t7_dir --file_training_params $hnm_training_output_dir/training_params.txt --saved_cand_ids $saved_cand_ids_file

if [ $? -eq 0 ]; then
   echo "T7 generation is done"  >> $pipeline_log_file
else
   echo "T7 generation has failed"  >> $pipeline_log_file
   exit
fi


# Run final inference and report the metrics on val 2 set

echo "Evaluating on the val 2 set"  >> $pipeline_log_file

cd $code_dir;

python run_benchmark_command_line_pipeline.py --biencoder_config $hnm_training_output_dir/config.json --biencoder_model $hnm_training_output_dir/pytorch_model.bin --entity_encoding $hnm_training_t7_dir/0_-1.t7 --biencoder_training_params $hnm_training_output_dir/training_params.txt --output_path $hnm_training_valid_2_prediction_dir --dataset_name val_2_set --test_file_path $validation_2_data_file --output_file_path $hnm_training_valid_2_prediction_dir/prediction.json --top_k 100 --output_score_file_path $hnm_training_valid_2_prediction_dir/scores.json --fast

if [ $? -eq 0 ]; then
   echo "Val 2 set evaluation is done"  >> $pipeline_log_file
else
   echo "Val 2 set evaluation has failed"  >> $pipeline_log_file
   exit
fi


# Run final inference and report the metrics on test set

echo "Evaluating on the test set"  >> $pipeline_log_file

cd $code_dir;

python run_benchmark_command_line_pipeline.py --biencoder_config $hnm_training_output_dir/config.json --biencoder_model $hnm_training_output_dir/pytorch_model.bin --entity_encoding $hnm_training_t7_dir/0_-1.t7 --biencoder_training_params $hnm_training_output_dir/training_params.txt --output_path $hnm_training_prediction_dir --dataset_name test_set --test_file_path $test_data_file --output_file_path $hnm_training_prediction_dir/prediction.json --top_k 100 --output_score_file_path $hnm_training_prediction_dir/scores.json --fast

if [ $? -eq 0 ]; then
   echo "Test set evaluation is done"  >> $pipeline_log_file
else
   echo "Test set evaluation has failed"  >> $pipeline_log_file
   exit
fi


#echo "Housekeeping- deleting the intermediate directories" >> $pipeline_log_file
#rm -rf $vanilla_training_data_dir
#rm -rf $vanilla_training_output_dir
#rm -rf $vanilla_training_t7_dir
#rm -rf $vanilla_training_prediction_dir
#rm -rf $hnm_training_data_dir


echo "Pipeline completed"  >> $pipeline_log_file

