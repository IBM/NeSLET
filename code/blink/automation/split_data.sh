#!/bin/bash -x

# arg1: training file
# arg2: val file
# arg3: base dir name. This script will create the directory
# arg4: dataset name. for example: blink_wiki, fget_wiki_um, fget_wiki_conll

training_data_file=$1
validation_data_file=$2
base_dir=$3
dataset_name=$4


for percentage_training_data_to_use in 0.01 0.1 1.0 5.0
do
  full_len=`wc -l $training_data_file | cut -d " " -f 1`
  lines_to_keep=`python -c "print(int( ($percentage_training_data_to_use * $full_len)/100 ))"`

  if [[ $percentage_training_data_to_use =  0.01 ]]
  then
     suffix=_0_01
  elif [[ $percentage_training_data_to_use =  0.1 ]]
  then
     suffix=_0_1
  elif [[ $percentage_training_data_to_use =  1.0 ]]
  then
     suffix=_1
  elif [[ $percentage_training_data_to_use =  5.0 ]]
  then
     suffix=_5
  else
     echo "Illegal value"
     exit
  fi

  output_dir_name=$base_dir/$dataset_name$suffix

  mkdir $output_dir_name

  head -n $lines_to_keep $training_data_file > $output_dir_name/train.jsonl
  cp $validation_data_file $output_dir_name/valid.jsonl


  echo "Created training and val sets. Took ${lines_to_keep} from ${training_data_file}"
done

