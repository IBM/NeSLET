import torch
import torch.nn as nn

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            #torch.nn.init.ones_(m.bias)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, model_type):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        if model_type == torch.IntTensor([1]):
            self.l1 = nn.Linear(input_size, 1,bias=False)
            #torch.nn.init.xavier_uniform_(self.l1.weight)

            #self.relu = nn.ReLU()
        if model_type == torch.IntTensor([2]):
            self.l1 = nn.Linear(input_size, 1,bias=True)


    def forward(self, x, model_type):
        if model_type == torch.IntTensor([1]):
            out = self.l1(x)
            #out = self.relu(out)
            #out = self.l2(out)
        if model_type == torch.IntTensor([2]):
            out = self.l1(x)
        return out

