import torch as th
import torch.nn as nn
import torch.nn.functional as F
from runnorm import RunNorm


def leaky_tanh(input, slope=0.01):
    return th.tanh(input * (2 / 3)) * 1.7159 + input * slope


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None, input_dropout=0.5, input_noise=1e-5,
                 recurrent_dropout=0.3, recurrent_noise=1e-3, activation=F.leaky_relu, **kwargs):
        super().__init__()
        if (hidden_size is None):
            hidden_size = 20*input_size

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_dropout = input_dropout
        self.input_noise = input_noise
        self.recurrent_dropout = recurrent_dropout
        self.recurrent_noise = recurrent_noise
        self.activation = activation
        
        #self.norm = RunNorm(num_features=input_size, momentum=momentum)
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 4, bias=True)
        self.rnn1 = nn.LSTMCell(input_size=hidden_size // 4, hidden_size=hidden_size // 8)
        self.fc3 = nn.Linear(hidden_size // 8, output_size, bias=True)
        
        # Initialize hidden state and cell state
        self.prev_state1 = None

    def reset(self):
        self.prev_state1 = None
        
    @property
    def sequential(self):
        return True
    
    def forward(self, inputs):
        one_sample_mode = False
        if inputs.dim() <= 2:
            one_sample_mode = True
            inputs = inputs.unsqueeze(dim=0)

        outputs = []
        for i in range(inputs.size(0)):
            x = inputs[i]
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

            if (self.recurrent_dropout is not None) and (self.prev_state1 is not None):
                h1, c1 = self.prev_state1
                self.prev_state1 = (
                    F.dropout(h1, p=self.recurrent_dropout, training=self.training),
                    F.dropout(c1, p=self.recurrent_dropout, training=self.training)
                )
            h1, c1 = self.rnn1(x, self.prev_state1)
            self.prev_state1 = (h1, c1)
            x = h1.clone()
            
            yhat = self.fc3(x)
            outputs.append(yhat)

        # Combine sequence of outputs
        if one_sample_mode:
            return yhat
        outputs = th.stack(outputs, dim=0)
        return outputs

    def extra_repr(self):
        return '''input_size={}, output_size={}, hidden_size={},
input_dropout={}, input_noise={},
recurrent_dropout={}, recurrent_noise={},
activation={}'''.format(self.input_size, self.output_size, self.hidden_size,
            self.input_dropout, self.input_noise, self.recurrent_dropout, self.recurrent_noise,
            str(self.activation))
