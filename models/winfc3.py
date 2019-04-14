import torch as th
import torch.nn as nn
import torch.nn.functional as F
from runnorm import RunNorm


def leaky_tanh(input, slope=0.01):
    return th.tanh(input * (2 / 3)) * 1.7159 + input * slope


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None, window_size=10, input_dropout=0.5, input_noise=0.1,
                 activation=F.leaky_relu, **kwargs):
        super().__init__()
        if (hidden_size is None):
            hidden_size = input_size*window_size // 2

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.input_dropout = input_dropout
        self.input_noise = input_noise
        self.activation = activation
        
        self.fc1 = nn.Linear(window_size*input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 4, bias=True)
        self.fc3 = nn.Linear(hidden_size // 4, output_size, bias=True)

        self.input_frames = []

    def reset(self):
        self.input_frames.clear()

    @property
    def sequential(self):
        return True

    def forward(self, inputs):
        one_sample_mode = False
        if (inputs.dim() <= 2):
            one_sample_mode = True
            inputs = inputs.unsqueeze(dim=0)
        
        outputs = []
        for i in range(inputs.size(0)):
            # Get next frame
            frame = inputs[i]
            
            # Apply dropout if needed
            if (self.input_dropout is not None):
                frame = F.dropout(frame, p=self.input_dropout, training=self.training)

            # Add white noise if needed (only at training time)
            if (self.input_noise is not None) and self.training:
                frame = frame + th.randn_like(frame) * self.input_noise

            # Get last frames of window_size
            self.input_frames.append(frame)
            if (len(self.input_frames) > self.window_size):
                self.input_frames.pop(0)
            
            if (len(self.input_frames) < self.window_size):
                continue
            
            x = th.stack(self.input_frames, dim=-1)
            x = x.view(x.size(0), -1)

            # Pass x through model
            x = self.fc1(x)
            x = self.activation(x)
            
            # Apply dropout if needed
            if (self.input_dropout is not None):
                x = F.dropout(x, p=(self.input_dropout/4), training=self.training)

            # Add white noise if needed (only at training time)
            if (self.input_noise is not None) and self.training:
                x = x + th.randn_like(x) * (self.input_noise / 2)
            
            x = self.fc2(x)
            x = self.activation(x)

            yhat = self.fc3(x)
            outputs.append(yhat)
            
        # Combine sequence of outputs
        outputs = th.stack(outputs, dim=0)
        if one_sample_mode:
            outputs = outputs.squeeze(dim=0)
        return outputs

    def extra_repr(self):
        return '''input_size={}, output_size={}, hidden_size={}, window_size={},
input_dropout={}, input_noise={},
activation={}'''.format(self.input_size, self.output_size, self.hidden_size,
            self.window_size, self.input_dropout, self.input_noise, str(self.activation))
