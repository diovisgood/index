import torch as th
import torch.nn as nn
import torch.nn.functional as F
#from runnorm import RunNorm


def leaky_tanh(input, slope=0.01):
    return th.tanh(input * (2 / 3)) * 1.7159 + input * slope


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None, input_dropout=0.5, input_noise=0.1,
                 activation=F.leaky_relu, **kwargs):
        super().__init__()
        if (hidden_size is None):
            hidden_size = 40*input_size

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_dropout = input_dropout
        self.input_noise = input_noise
        self.activation = activation
        
        #self.norm = RunNorm(num_features=input_size, momentum=momentum)
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2, bias=True)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4, bias=True)
        self.fc4 = nn.Linear(hidden_size // 4, hidden_size // 8, bias=True)
        self.fc5 = nn.Linear(hidden_size // 8, output_size, bias=True)

    def reset(self):
        pass

    def forward(self, x):
        # Pass x through model
        
        # Apply dropout if needed
        if (self.input_dropout is not None):
            x = F.dropout(x, p=self.input_dropout, training=self.training)

        # Add white noise if needed (only at training time)
        if self.training and (self.input_noise is not None):
            x = x + th.randn_like(x)*self.input_noise
            
        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.activation(x)

        x = self.fc4(x)
        x = self.activation(x)

        yhat = self.fc5(x)
        return yhat

    def extra_repr(self):
        return '''input_size={}, output_size={}, hidden_size={},
input_dropout={}, input_noise={},
activation={}'''.format(self.input_size, self.output_size, self.hidden_size,
            self.input_dropout, self.input_noise, str(self.activation))
