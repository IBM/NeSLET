import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import NeuralNet
import sys
import argparse
import numpy as np
from data_loader import Val_Dataset
import utils
import os
np.random.seed(0)

device = "cpu"
ignore_features = []
parser = argparse.ArgumentParser()
parser.add_argument("--data_path",dest="data_path",type=str,help="Path to training data.")
parser.add_argument("--model_type",default="model_1",type=str,help="Which Model to use." )
parser.add_argument("--model_path",dest="model_path",type=str,help="Path to models.")

args = parser.parse_args()

checkpoint_dir = os.path.join(args.model_path,args.model_type)
checkpoint_file_path = os.path.join(checkpoint_dir,'checkpoint.pth')

def load_checkpoint(model,filename):
    print("Loading Checkpoint")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def calculate_val_accuracy(model,model_type,val_loader,device="cpu"):
    no_of_correct_predictions = 0
    for i, data in enumerate(val_loader):
        #print(data.shape)
        postive_data = data[0][0]
        #print(postive_data)
        postive_data = postive_data.to(device)
        postive_score = model(postive_data.float(),model_type)
        #print(postive_score.item())
        neg_scores = []
        #print(len(data[0]))
        for i in range(1,len(data[0])):
            #print(i)
            negative_data = data[0][i]
            #print(negative_data)
            negative_data = negative_data.to(device)
            score = model(negative_data.float(), model_type)
            neg_scores.append(score.item())
        #print(neg_scores)
        if postive_score > max(neg_scores):
            no_of_correct_predictions+=1
    accuracy = no_of_correct_predictions/len(val_loader)
    return accuracy


def prepare_data_val(input_data):
    output_data = []
    output_labels = []
    labels = input_data[:, -1].tolist()
    index_postive = labels.index(1.0)
    all_examples = list(range(len(input_data)))
    list_of_negative_examples = [x for x in all_examples if x != index_postive]

    positve_features = input_data[index_postive, :-1].tolist()
    positve_features = [i for j, i in enumerate(positve_features) if j not in ignore_features]
    new_example = []
    new_example.append(positve_features)
    for item in list_of_negative_examples:
        negative_features = input_data[item, :-1].tolist()
        negative_features = [i for j, i in enumerate(negative_features) if j not in ignore_features]
        new_example.append(negative_features)


    return new_example

data = torch.load(args.data_path)
random_permuation = list(np.random.permutation(len(data)))
train_split_size = int(0.8*len(data))
training_indexes = random_permuation[:train_split_size]
validation_indexes = random_permuation[train_split_size:]
val_data = data[validation_indexes,:,:]


count_zero_labels_val = 0
modified_validation_data = []
for i in range(len(val_data)):
    sum = torch.sum(val_data[i, :, -1]).item()
    if sum == 0.0:
        count_zero_labels_val += 1
    else:
        temp_data = val_data[i]
        temp_val_data = prepare_data_val(val_data[i])
        modified_validation_data.append(temp_val_data)

input_size = len(modified_validation_data[0][0])



temp_model_type = int(args.model_type.split('_')[1])
int_model_type = torch.IntTensor([temp_model_type])
model = NeuralNet(input_size,int_model_type).to(device)
model = load_checkpoint(model,checkpoint_file_path)
model.eval()
validation_set = Val_Dataset(modified_validation_data)
val_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=1)

val_accuracy = calculate_val_accuracy(model, int_model_type, val_loader)
print("Validation set accuracy {}".format(val_accuracy))

