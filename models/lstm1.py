import torch as th
import torch.nn as nn
import torch.nn.functional as F
from runnorm import RunNorm


def leaky_tanh(input, slope=0.01):
    return th.tanh(input * (2 / 3)) * 1.7159 + input * slope


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None, input_dropout=0.6, input_noise=1e-5,
                 recurrent_dropout=0.5, recurrent_noise=1e-5, activation=leaky_tanh, **kwargs):
        super().__init__()
        if (hidden_size is None):
            hidden_size = input_size // 2

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_dropout = input_dropout
        self.input_noise = input_noise
        self.recurrent_dropout = recurrent_dropout
        self.recurrent_noise = recurrent_noise
        self.activation = activation
        
        #self.norm = RunNorm(num_features=input_size, momentum=momentum)
        self.rnn1 = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.fc1 = nn.Linear(hidden_size, output_size, bias=True)

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
            
            # Apply input dropout if needed
            if (self.input_dropout is not None):
                x = F.dropout(x, p=self.input_dropout, training=self.training)
    
            # Add white noise to input if needed (only at training time)
            if self.training and (self.input_noise is not None):
                x = x + th.randn_like(x)*self.input_noise
                
            # Apply recurrent dropout if needed
            if (self.recurrent_dropout is not None) and (self.prev_state1 is not None):
                h1, c1 = self.prev_state1
                self.prev_state1 = (
                    F.dropout(h1, p=self.recurrent_dropout, training=self.training),
                    F.dropout(c1, p=self.recurrent_dropout, training=self.training)
                )

            # Add white noise to recurrent if needed (only at training time)
            if self.training and (self.recurrent_noise is not None) and (self.prev_state1 is not None):
                h1, c1 = self.prev_state1
                self.prev_state1 = (
                    h1 + th.randn_like(h1)*self.recurrent_noise,
                    c1 + th.randn_like(c1)*self.recurrent_noise,
                )
            h1, c1 = self.rnn1(x, self.prev_state1)
            self.prev_state1 = (h1, c1)
            x = h1.clone()
            
            yhat = self.fc1(x)
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
            self.input_dropout, self.input_noise, self.recurrent_dropout, self.recurrent_noise, str(self.activation))
