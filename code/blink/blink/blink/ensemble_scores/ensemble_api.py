import torch
import torch.nn as nn
from model import NeuralNet


def load_checkpoint(filename,device):
    print("Loading Checkpoint")
    checkpoint = torch.load(filename,map_location=device)
    state_dict = checkpoint['state_dict']
    int_model_type = checkpoint['model_type']
    return state_dict,int_model_type


def load_model(input_size,checkpoint_file_path,device = 'cpu'):
    # temp_model_type = int(model_type.split('_')[1])
    # int_model_type = torch.IntTensor([temp_model_type])
    state_dict, int_model_type= load_checkpoint(checkpoint_file_path,device)
    model = NeuralNet(input_size, int_model_type).to(device)
    model.load_state_dict(state_dict)
    return model,int_model_type

def get_ensemble_score(model,int_model_type,input,device = 'cpu'):
    score = model(torch.FloatTensor(input), int_model_type)
    return score.item()



