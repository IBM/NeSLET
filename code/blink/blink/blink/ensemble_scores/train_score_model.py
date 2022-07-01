import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import NeuralNet,weights_init
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
from data_loader import Dataset,Val_Dataset
import utils

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",dest="data_path",type=str,help="Path to training data.")
parser.add_argument("--learning_rate",default=1e-4,type=float,help="The initial learning rate for Adam.")
parser.add_argument("--training_split_size",default=0.8,type=float,help="train split size")
parser.add_argument("--epochs",default=4,type=int,help="train epochs" )
parser.add_argument("--batch_size",default=32,type=int,help="train epochs" )
parser.add_argument("--model_type",default="model_1",type=str,help="Which Model to use." )
parser.add_argument("--margin",default=1,type=int,help="margin in MarginRankingLoss" )
parser.add_argument("--output_path",dest="output_path",type=str,help="Output Path.")


#ignore_features = [4,5,6,7,8,11]
ignore_features = []

args = parser.parse_args()

input_file_name = os.path.basename(args.data_path)
if input_file_name.find('tempqa') != -1:
    dataset_name = 'tempqa'
elif input_file_name.find('swq') != -1:
    dataset_name = 'swq'
else:
    print("Error")
    sys.exit("Dataset is not correct")

if input_file_name.find('100') != -1:
    model_percentage = '100'
elif input_file_name.find('5') != -1:
    model_percentage = '5'
else:
    print("Error")
    sys.exit("Model percentage is not correct")

#output directory path
output_path = os.path.join(args.output_path, dataset_name)
output_path = os.path.join(output_path,'model_'+model_percentage)
model_type =args.model_type

output_path = os.path.join(output_path,model_type)
logger = utils.get_logger(output_path)
logger.info("Training data file path {}".format(args.data_path))
logger.info("model type {}".format(model_type))
############## TENSORBOARD ########################
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter(os.path.join(output_path,'runs'))
###################################################

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_checkpoint(state,filename):
    logger.info("Saving Checkpoint")
    torch.save(state,filename)

def calculate_val_accuracy(model,model_type,val_loader):
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

def prepare_data(input_data):
    output_data = []
    output_labels = []
    labels = input_data[:, -1].tolist()
    index_postive = labels.index(1.0)
    all_examples = list(range(len(input_data)))
    list_of_negative_examples = [x for x in all_examples if x != index_postive]

    positve_features = input_data[index_postive, :-1].tolist()
    positve_features = [i for j, i in enumerate(positve_features) if j not in ignore_features]
    for item in list_of_negative_examples:
        new_example = []
        negative_features = input_data[item, :-1].tolist()
        negative_features = [i for j, i in enumerate(negative_features) if j not in ignore_features]

        new_example.append(positve_features)
        new_example.append(negative_features)
        output_data.append(new_example)
        output_labels.append(1)


    return output_data,output_labels



data = torch.load(args.data_path)
random_permuation = list(np.random.permutation(len(data)))
train_split_size = int(args.training_split_size*len(data))
training_indexes = random_permuation[:train_split_size]
validation_indexes = random_permuation[train_split_size:]
train_data = data[training_indexes,:,:]
val_data = data[validation_indexes,:,:]

# Hyper-parameters
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
margin = args.margin

logger.info("No. of epochs for training {}".format(num_epochs))
logger.info("batch_size {}".format(batch_size))
logger.info("learning_rate {}".format(learning_rate))
logger.info("margin for max-margin training {}".format(margin))

logger.info("No of examples in original training data {}".format(len(train_data)))






count_zero_labels_train = 0
modified_training_data  = []
modified_training_labels  = []

for i in range(len(train_data)):
    sum = torch.sum(train_data[i, :, -1]).item()
    if sum == 0.0:
        count_zero_labels_train += 1
    else:
        temp_data = train_data[i]
        temp_training_data, temp_training_labels = prepare_data(train_data[i])
        modified_training_data.extend(temp_training_data)
        modified_training_labels.extend(temp_training_labels)

input_size = len(modified_training_data[0][0])

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


logger.info("No. input Features {}".format(input_size))
logger.info("No of examples without gold entities in train data {}".format(count_zero_labels_train))
logger.info("No of examples in Modified training data {}".format(len(modified_training_data)))
logger.info("No of examples in Modified val data {}".format(len(modified_validation_data)))

# Data loader
training_set = Dataset(modified_training_data, modified_training_labels)
train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=False, num_workers=1)

validation_set = Val_Dataset(modified_validation_data)
val_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=1)






temp_model_type = int(model_type.split('_')[1])
int_model_type = torch.IntTensor([temp_model_type])


model = NeuralNet(input_size,int_model_type).to(device)
model.apply(weights_init)



# Loss and optimizer
criterion = nn.MarginRankingLoss(margin=margin)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

logger.info("Learnable parameters")
for name, param in model.named_parameters():
    if param.requires_grad:
        logger.info("{} {}".format(name,param.data))
        #logger.info("{}".format(name))


# Train the model
running_loss = 0.0
val_accuracy = 0.0
n_total_steps = len(train_loader)
model.train()
for epoch in range(num_epochs):
    for i, (postive_data, negative_data, labels) in enumerate(train_loader):

        postive_data = postive_data.to(device)
        negative_data = negative_data.to(device)
        labels = labels.to(device)
        #print(type(postive_data))
        # Forward pass
        postive_scores = model(postive_data.float(),int_model_type)
        #print(postive_scores)
        negative_scores = model(negative_data.float(), int_model_type)
        #print(negative_scores)
        loss = criterion(postive_scores, negative_scores,labels)
        #print(loss)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    logger.info (f'Epoch [{epoch+1}/{num_epochs}]], Loss: {running_loss/len(training_set):.6f}')
    ############## TENSORBOARD ########################
    writer.add_scalar('training loss', running_loss/len(training_set), epoch)
    model.eval()
    new_val_accuracy = calculate_val_accuracy(model,int_model_type,val_loader)
    logger.info("Validation set accuracy {}".format(new_val_accuracy))
    if new_val_accuracy > val_accuracy:
        val_accuracy = new_val_accuracy
        checkpoint = {'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'best_epoch':epoch, 'best_val_acc':val_accuracy,'model_type':int_model_type}

        model_file_name = os.path.join(output_path,'checkpoint.pth')
        save_checkpoint(checkpoint,model_file_name)
    running_loss = 0.0
    model.train()
    ###################################################


    ###################################################

logger.info("Learned parameters")
for name, param in model.named_parameters():
    if param.requires_grad:
        logger.info("{} {}".format(name,param.data))
        #logger.info("{}".format(name))