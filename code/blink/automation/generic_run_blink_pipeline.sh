#!/bin/bash -x

#SBATCH -J test_job     # overridden by commandline argument
#SBATCH -o /scratch/job_logs/stdout_%j # overridden by commandline argument
#SBATCH -e /scratch/job_logs/stderr_%j # overridden by commandline argument
#SBATCH --qos=dcs-48hr
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1


# Pick these variables from the environment or uncomment and initialize them below

# something like /ZEL/code/blink/blink
code_dir=$1

# place to store all the outputs
experiment_dir=$2

# some number between 0 to 100
percentage_training_data_to_use=$3

num_vanilla_training_epochs=$4
num_hnm_training_epochs=$5

path_to_model=$6
bert_model=$7  # either bert-base-uncased or bert-large-uncased

training_data_file=$8
validation_data_file=$9
validation_2_data_file=${10}
test_data_file=${11}

vanilla_train_batch_size=${12}
hnm_train_batch_size=${13}

entities_file=${14}
saved_cand_ids_file=${15}

vanilla_train_lr=${16}
hnm_train_lr=${17}

#------------------
# Aimos
#------------------

#entities_file=/scratch-shared/facebook_original_models/entity.jsonl
#saved_cand_ids_file=/scratch-shared/entity_token_ids_128.t7


# create the directories inside $experiment_dir

mkdir $experiment_dir
vanilla_training_data_dir=${experiment_dir}/vanilla_training_data_dir ; mkdir $vanilla_training_data_dir
vanilla_training_output_dir=${experiment_dir}/vanilla_training_output_dir ; mkdir $vanilla_training_output_dir
vanilla_training_t7_dir=${experiment_dir}/vanilla_training_t7_dir ; mkdir $vanilla_training_t7_dir
vanilla_training_prediction_dir=${experiment_dir}/vanilla_training_prediction_dir ; mkdir $vanilla_training_prediction_dir
hnm_training_data_dir=${experiment_dir}/hnm_training_data_dir ; mkdir $hnm_training_data_dir
hnm_training_output_dir=${experiment_dir}/hnm_training_output_dir ; mkdir $hnm_training_output_dir
hnm_training_t7_dir=${experiment_dir}/hnm_training_t7_dir ; mkdir $hnm_training_t7_dir
hnm_training_valid_prediction_dir=${experiment_dir}/hnm_training_valid_prediction_dir ; mkdir $hnm_training_valid_prediction_dir
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

cd $code_dir ; PYTHONPATH=. python blink/biencoder/train_biencoder.py --data_path $vanilla_training_data_dir --output_path $vanilla_training_output_dir --learning_rate $vanilla_train_lr --train_batch_size $vanilla_train_batch_size --eval_batch_size 32 --num_train_epochs $num_vanilla_training_epochs --data_parallel --bert_model $bert_model --max_context_length 64 --eval_interval 9999999 --save_interval 10000 --zeshel False --lowercase --print_interval 2000 --shuffle True --path_to_model $path_to_model

if [ $? -eq 0 ]; then
   echo "Vanilla training done"  >> $pipeline_log_file
else
   echo "Vanilla training has failed"  >> $pipeline_log_file
   exit
fi


# T7 generation

echo "Starting T7 generation"  >> $pipeline_log_file

cd $code_dir ;
python generate_candidates.py --path_to_model_config $vanilla_training_output_dir/config.json --path_to_model $vanilla_training_output_dir/pytorch_model.bin --entity_dict_path $entities_file --encoding_save_file_dir $vanilla_training_t7_dir --file_training_params $vanilla_training_output_dir/training_params.txt --saved_cand_ids $saved_cand_ids_file --batch_size 480


if [ $? -eq 0 ]; then
   echo "T7 generation done"  >> $pipeline_log_file
else
   echo "T7 generation has failed"  >> $pipeline_log_file
   exit
fi


# Generate prediction of training data by running inference code

echo "Generating predictions on training set"  >> $pipeline_log_file

cd $code_dir;

python run_benchmark_command_line_pipeline.py --biencoder_config $vanilla_training_output_dir/config.json --biencoder_model $vanilla_training_output_dir/pytorch_model.bin --entity_encoding $vanilla_training_t7_dir/0_-1.t7 --biencoder_training_params $vanilla_training_output_dir/training_params.txt --output_path $vanilla_training_prediction_dir --dataset_name training_set --test_file_path $vanilla_training_data_dir/train.jsonl --output_file_path $vanilla_training_prediction_dir/prediction.json --fast --top_k 10 --output_score_file_path $vanilla_training_prediction_dir/scores.json --entity_catalogue $entities_file


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

PYTHONPATH=. python blink/biencoder/train_biencoder.py --data_path $vanilla_training_data_dir --output_path $hnm_training_output_dir --learning_rate $hnm_train_lr --train_batch_size $hnm_train_batch_size --eval_batch_size 128 --num_train_epochs $num_hnm_training_epochs --data_parallel --bert_model $bert_model --max_context_length 64 --eval_interval 9999999 --save_interval 10000 --zeshel False --lowercase --print_interval 2000 --shuffle True --entities_file $entities_file --hard_negatives_file $hnm_training_data_dir/topk.jsonl --max_num_negatives 9 --path_to_model $vanilla_training_output_dir/pytorch_model.bin


if [ $? -eq 0 ]; then
   echo "hnm training is done"  >> $pipeline_log_file
else
   echo "hnm training has failed"  >> $pipeline_log_file
   exit
fi


# Generate t7 file for hnm model

echo "Generating T7"  >> $pipeline_log_file

cd $code_dir;
python generate_candidates.py --path_to_model_config $hnm_training_output_dir/config.json --path_to_model $hnm_training_output_dir/pytorch_model.bin --entity_dict_path $entities_file --encoding_save_file_dir $hnm_training_t7_dir --file_training_params $hnm_training_output_dir/training_params.txt --saved_cand_ids $saved_cand_ids_file --batch_size 480

if [ $? -eq 0 ]; then
   echo "T7 generation is done"  >> $pipeline_log_file
else
   echo "T7 generation has failed"  >> $pipeline_log_file
   exit
fi

#=====================================================================================
# Run final inference and report the metrics on val set

echo "Evaluating on the val set"  >> $pipeline_log_file

cd $code_dir;

python run_benchmark_command_line_pipeline.py --biencoder_config $hnm_training_output_dir/config.json --biencoder_model $hnm_training_output_dir/pytorch_model.bin --entity_encoding $hnm_training_t7_dir/0_-1.t7 --biencoder_training_params $hnm_training_output_dir/training_params.txt --output_path $hnm_training_valid_prediction_dir --dataset_name val_set --test_file_path $validation_data_file --output_file_path $hnm_training_valid_prediction_dir/prediction.json --top_k 100 --output_score_file_path $hnm_training_valid_prediction_dir/scores.json --fast --entity_catalogue $entities_file

if [ $? -eq 0 ]; then
   echo "Val set evaluation is done"  >> $pipeline_log_file
else
   echo "Val set evaluation has failed"  >> $pipeline_log_file
   exit
fi

#=====================================================================================
# Run final inference and report the metrics on val 2 set

echo "Evaluating on the val 2 set"  >> $pipeline_log_file

cd $code_dir;

python run_benchmark_command_line_pipeline.py --biencoder_config $hnm_training_output_dir/config.json --biencoder_model $hnm_training_output_dir/pytorch_model.bin --entity_encoding $hnm_training_t7_dir/0_-1.t7 --biencoder_training_params $hnm_training_output_dir/training_params.txt --output_path $hnm_training_valid_2_prediction_dir --dataset_name val_2_set --test_file_path $validation_2_data_file --output_file_path $hnm_training_valid_2_prediction_dir/prediction.json --top_k 100 --output_score_file_path $hnm_training_valid_2_prediction_dir/scores.json --fast --entity_catalogue $entities_file

if [ $? -eq 0 ]; then
   echo "Val 2 set evaluation is done"  >> $pipeline_log_file
else
   echo "Val 2 set evaluation has failed"  >> $pipeline_log_file
   exit
fi


#=====================================================================================
# Run final inference and report the metrics on test set

echo "Evaluating on the test set"  >> $pipeline_log_file

cd $code_dir;

python run_benchmark_command_line_pipeline.py --biencoder_config $hnm_training_output_dir/config.json --biencoder_model $hnm_training_output_dir/pytorch_model.bin --entity_encoding $hnm_training_t7_dir/0_-1.t7 --biencoder_training_params $hnm_training_output_dir/training_params.txt --output_path $hnm_training_prediction_dir --dataset_name test_set --test_file_path $test_data_file --output_file_path $hnm_training_prediction_dir/prediction.json --top_k 100 --output_score_file_path $hnm_training_prediction_dir/scores.json --fast --entity_catalogue $entities_file

if [ $? -eq 0 ]; then
   echo "Test set evaluation is done"  >> $pipeline_log_file
else
   echo "Test set evaluation has failed"  >> $pipeline_log_file
   exit
fi

#=====================================================================================


echo "Housekeeping- deleting the intermediate directories" >> $pipeline_log_file
rm -rf $vanilla_training_data_dir
rm -rf $vanilla_training_output_dir
rm -rf $vanilla_training_t7_dir
rm -rf $vanilla_training_prediction_dir
rm -rf $hnm_training_data_dir


echo "Pipeline completed"  >> $pipeline_log_file

