import torch
import numpy as np
from ensemble_api import load_model,get_ensemble_score
np.random.seed(0)

data = torch.load('zel_everything/ZEL/data_dinesh/tempqa/tempqa_val_set_full_blink_prediction_model_100_with_wikidata.t7')
random_permuation = list(np.random.permutation(len(data)))
train_split_size = int(0.8*len(data))
training_indexes = random_permuation[:train_split_size]
validation_indexes = random_permuation[train_split_size:]
val_data = data[validation_indexes,:,:]

checkpoint_file_path ='zel_everything/models_ensembles/tempqa/model_100/model_2/checkpoint.pth'

# load_model function will load the trained model
# argument 1 is no. of input scores used
# argument 2 is path to trained model
# argument 3 is optional says whether one wants to use 'cpu' or 'gpu'

model,model_type = load_model(12,checkpoint_file_path, 'cpu')

# get_ensemble_score function return the ensemble score
# argument 1 and 2 are output returned by load_model function
# argument 3 list of scores.
# argument 4 is optional says whether one wants to use 'cpu' or 'gpu'

score = get_ensemble_score(model,model_type, val_data[0][0][:-1].tolist(),'cpu')
print(score)