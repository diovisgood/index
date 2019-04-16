import os
import math
import copy
import torch as th
import torch.nn as nn
import pandas as pd
import arrow
import argparse
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import data

Config = None

Params = dict(
    model='fc5',
    preprocess=None,
    target_offset=1,
    sequence_size=128,
    batch_size=256,
    learning_rate=0.000003,
    weight_decay=30,
    early_stop=0.3
)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Index')
    parser.add_argument('config', type=str, default='MICEX10',
        help='Name of index subdirectory which contains configuration and data to process.')
    parser.add_argument('-model', type=str, default=Params['model'],
        help='Name of model to train. Choose one from models subdirectory.')
    parser.add_argument('-preprocess', type=str, default=Params['preprocess'],
        help='Choose how to preprocess prices: none, diff or logret, default: diff')
    parser.add_argument('-target-offset', type=int, default=Params['target_offset'],
        help='How many future intervals to predict, int, default: 1')
    parser.add_argument('-sequence-size', type=int, default=Params['sequence_size'],
        help='Sequence length, int')
    parser.add_argument('-batch-size', type=int, default=Params['batch_size'],
        help='Batch size, int')
    parser.add_argument('-learning-rate', type=float, default=Params['learning_rate'],
        help='Learning rate, float (0...1)')
    parser.add_argument('-weight-decay', type=float, default=Params['weight_decay'],
        help='Weight decay, float [0...1)')
    parser.add_argument('-early-stop', type=float, default=Params['early_stop'],
        help='Early stopping threshold, float (0...1)')
    parser.add_argument('-log-interval', type=int, default=5,
        help='Print log interval in seconds, int, default: 5')
    parser.add_argument('-autosave-interval', type=int, default=60,
        help='Autosave interval in seconds, int, default: 60')
    args = parser.parse_args()
    preprocess = args.preprocess
    target_offset = args.target_offset
    sequence_size = args.sequence_size
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    early_stop = args.early_stop
    
    # Setup appropriate scaler depending on preprocess argument
    scaler = StandardScaler(with_mean=False, with_std=True) if (preprocess == 'diff') or (preprocess == 'logret')\
        else MinMaxScaler()

    # Setup torch
    th.set_num_threads(2)
    th.set_default_dtype(th.float32)
    th.set_default_tensor_type(th.FloatTensor)
    
    # Load configuration
    global Config
    Config = __import__(args.config, globals(), locals())
    assert hasattr(Config, 'Dataset_File')
    assert hasattr(Config, 'Input_Size')
    
    # Initialize model
    print('Initializing model')
    _module = __import__('models.' + args.model, globals(), locals(), fromlist=['Model'], level=0)
    model_class = _module.Model
    model = model_class(input_size=Config.Input_Size, output_size=1, hidden_size=1024)
    
    # Construct file names
    name = '{}_tar{}_{}_hid{}_lr{}_wd{}_bs{}_dr{}_nz{}{}{}'.format(args.model, target_offset, preprocess,
        model.hidden_size, learning_rate, weight_decay, batch_size, model.input_dropout, model.input_noise,
        ('_rdr' + str(model.recurrent_dropout) if hasattr(model, 'recurrent_dropout') else ''),
        ('_rnz' + str(model.recurrent_noise) if hasattr(model, 'recurrent_noise') else '')
    )
    model_file_name = args.config + '/' + name + '.pt'
    stat_file_name = args.config + '/' + name + '.csv'
    plot_file_name = args.config + '/' + name + '.png'
    train_chart_file_name = args.config + '/' + name + '.train.png'
    test_chart_file_name = args.config + '/' + name + '.test.png'
    stat = None
    
    # Try to load model parameters
    if os.path.isfile(model_file_name):
        # Load model parameters
        state_dict = th.load(model_file_name)
        model.load_state_dict(state_dict, strict=True)
        print(' Loaded model parameters from {}'.format(model_file_name))
        
        # Load statistics
        if os.path.isfile(stat_file_name):
            stat = pd.read_csv(stat_file_name, sep=';',
                dtype={'niter': int, 'epoch': int, 'train_loss': float, 'test_loss': float, 'min_test_loss': float})
            print('Loaded statistics from {}'.format(stat_file_name))
    else:
        print(' Could not find file to load model parameters: {}'.format(model_file_name))
    print(model)
    
    # Check if model supports sequential inputs or not
    if (not hasattr(model, 'sequential')) or (not model.sequential):
        sequence_size = None
        
    # Load dataset
    print('Loading dataset from {}'.format(Config.Dataset_File))
    dataset = data.loadDataset(Config.Dataset_File)
    print(dataset.head(10))
    
    # Preprocess dataset
    print('Preprocessing dataset, method={}, target_offset={}'.format(preprocess, target_offset))
    data.preprocessDataset(dataset, method=preprocess, target_offset=target_offset)
    print(dataset.head(10))
    
    # Save min, max of target values to denormalize NN output back to prices
    target = dataset.iloc[:, -1]
    target_min, target_max, target_mean, target_std = target.min(), target.max(), target.mean(), target.std()
    del target
    print('Target min={} max={} mean={} std={}\n'.format(target_min, target_max, target_mean, target_std))
    
    print('Splitting dataset into train and test')
    train, test = data.splitDataset(dataset, train_test_ratio=0.8, interval='month')
    
    print('Normalizing train and test datasets using method={}'.format(str(scaler)))
    data.normalizeDataset(train, scaler=scaler)
    data.normalizeDataset(test, scaler=scaler)
    
    train_size, test_size = len(train), len(test)
    print(' Train: total {} records'.format(train_size))
    print(train.head(10))
    print(' Test: total {} records'.format(test_size))
    print(test.head(10))

    # Prepare statistics table if not loaded yet
    if (stat is None):
        stat = pd.DataFrame(columns=('niter', 'epoch', 'train_loss', 'test_loss', 'min_test_loss'))
        stat.niter = stat.niter.astype('int32')
        stat.epoch = stat.epoch.astype('int32')

    # Initialize global iteration counter and epoch counter
    niter = int(stat.niter.iloc[-1]) if (len(stat) > 0) else 1
    epoch = int(stat.epoch.iloc[-1]) if (len(stat) > 0) else 1
    test_interval = max(1, int(train_size / test_size))

    # Initialize variables to track error for train and test datasets
    train_loss = float(stat.train_loss.iloc[-1]) if (len(stat) > 0) else None
    test_loss = float(stat.test_loss.iloc[-1]) if (len(stat) > 0) else None
    momentum = 2 / 100
    
    # We'll keep track of minimal test_loss and best model parameters
    min_test_loss = stat.min_test_loss.iloc[-1] if (len(stat) > 0) else None
    best_state_dict = copy.deepcopy(model.state_dict()) if (min_test_loss is not None) else None
    best_state_is_not_saved = False

    # Initialize optimizer
    criterion = nn.MSELoss(reduction='mean')
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # These variables are used to draw price and prediction charts for train and test datasets
    last_test_y, last_test_yhat = None, None
    last_train_y, last_train_yhat = None, None

    # Main cycle for epochs
    last_log_time = arrow.now()
    last_save_time = arrow.now()
    print('\nStarting training: lr={}, wd={}, sequence_size={}, batch_size={}, min_test_loss={}'.format(
        learning_rate, weight_decay, sequence_size, batch_size, min_test_loss))
    while (epoch <= 2000):
        # Cycle iterations for current epoch
        for x, y in data.iterateDataset(train, batch_size, sequence_size):
            # Convert numpy.ndarray to torch.Tensor
            x = th.tensor(x, dtype=th.get_default_dtype())
            y = th.tensor(y, dtype=th.get_default_dtype())
            
            # Run input through model and get prediction
            model.train()
            model.reset()
            yhat = model.forward(x)
            
            # Model may return output sequence that is shorter than input.
            # Especially if they work in 'windowed' mode.
            # We need to cut earliest timesteps from y, to make y and yhat the same sequence size.
            if (sequence_size is not None) and (y.size(0) > yhat.size(0)):
                y = y[-yhat.size(0):]
                
            # Calculate loss
            loss = criterion(yhat, y)
            loss = loss.sum()
            
            # Update mean train error
            train_loss = (train_loss * (1 - momentum) + loss.item() * momentum) \
                if (train_loss is not None) else loss.item()
            
            # Zero parameters gradient
            optimizer.zero_grad()
            # Calculate parameters gradient
            loss.backward()
            # Do weight decay. I don't use Adam's internal weight decay as it is implemented incorrectly.
            weightDecay(optimizer.param_groups, weight_decay)
            # Update model weights
            optimizer.step()
            # Check and fix NaNs or Inf in parameters
            checkWeights(optimizer.param_groups)
            
            # Sometimes perform validation test
            if (niter % test_interval == 0):
                test_loss, last_test_y, last_test_yhat = validate(test, model, criterion)
                last_test_y = denormalize(last_test_y, target_min, target_max, target_std, preprocess)
                last_test_yhat = denormalize(last_test_yhat, target_min, target_max, target_std, preprocess)
                # Update best results if needed
                if (min_test_loss is None) or (min_test_loss > test_loss):
                    min_test_loss = test_loss
                    best_state_dict = copy.deepcopy(model.state_dict())
                    best_state_is_not_saved = True
                    stat.loc[len(stat)] = [niter, epoch, train_loss, test_loss, min_test_loss]
                # Check for early-stopping
                if (test_loss - min_test_loss >= early_stop * min_test_loss):
                    print('Early stopping')
                    break
                    
            # Print out some log with log_interval
            if (arrow.now().timestamp - last_log_time.timestamp >= args.log_interval):
                print(' niter={} epoch={} train_loss={} test_loss={} min_test_loss={}'.format(
                    niter, epoch, train_loss, test_loss, min_test_loss))
                stat.loc[len(stat)] = [niter, epoch, train_loss, test_loss, min_test_loss]
                last_log_time = arrow.now()
            
            # Save model, stat and plot with autosave_interval
            if (arrow.now().timestamp - last_save_time.timestamp >= args.autosave_interval):
                if (best_state_dict is not None) and best_state_is_not_saved:
                    print('Saving model to {}'.format(model_file_name))
                    th.save(best_state_dict, model_file_name)
                    best_state_is_not_saved = False
                # Save statistics
                print('Saving stat, plot and charts to {}.*'.format(name))
                stat.to_csv(stat_file_name, sep=';', index=False)
                savePlot(plot_file_name, df=stat[['train_loss', 'test_loss']], title=args.model)
                # Validate model on train subset
                _, last_train_y, last_train_yhat = validate(train.head(len(test)), model, criterion)
                last_train_y = denormalize(last_train_y, target_min, target_max, target_std, preprocess)
                last_train_yhat = denormalize(last_train_yhat, target_min, target_max, target_std, preprocess)
                # Draw charts with real and predicted data
                saveChart(train_chart_file_name, last_train_y, last_train_yhat)
                saveChart(test_chart_file_name, last_test_y, last_test_yhat)
                last_save_time = arrow.now()
            
            # Increment iteration counter
            niter += 1

        # Check for early-stopping
        if (min_test_loss is not None) and (test_loss - min_test_loss >= early_stop * min_test_loss):
            break

        # Increment epoch counter
        epoch += 1

    # Load best model state
    model.load_state_dict(best_state_dict, strict=True)
    print('Saving best model to {}.*'.format(name))
    th.save(model.state_dict(), model_file_name)

    # Save statistics
    print('Saving stat to {}.csv'.format(name))
    stat.to_csv(stat_file_name, sep=';', index=False)
    savePlot(plot_file_name, df=stat[['train_loss', 'test_loss']], title=args.model)

    # Make final validation
    test_loss, last_test_y, last_test_yhat = validate(test, model, criterion)
    last_test_y = denormalize(last_test_y, target_min, target_max, target_std, preprocess)
    last_test_yhat = denormalize(last_test_yhat, target_min, target_max, target_std, preprocess)

    train_loss, last_train_y, last_train_yhat = validate(train.head(len(test)), model, criterion)
    last_train_y = denormalize(last_train_y, target_min, target_max, target_std, preprocess)
    last_train_yhat = denormalize(last_train_yhat, target_min, target_max, target_std, preprocess)

    saveChart(train_chart_file_name, last_train_y, last_train_yhat)
    saveChart(test_chart_file_name, last_test_y, last_test_yhat)
    print('Final results:')
    print(' train_loss={} test_loss={} min_test_loss={}'.format(train_loss, test_loss, min_test_loss))


def validate(dataset, model, criterion):
    model.eval()
    model.reset()
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1:].values
    x = th.tensor(x, dtype=th.get_default_dtype())
    y = th.tensor(y, dtype=th.get_default_dtype())
    # Add batch dimension if needed
    if hasattr(model, 'sequential') and model.sequential:
        x = x.unsqueeze(dim=1)
        y = y.unsqueeze(dim=1)
    # Pass whole sequence through model. Get prediction
    yhat = model.forward(x)
    # Cut down y to be the same length as yhat as some windowed models may return a shorter sequence
    if (y.size(0) > yhat.size(0)):
        y = y[-yhat.size(0):]
    # Calculate loss
    loss = criterion(yhat, y)
    loss = loss.sum()
    # loss = calculateLoss(yhat, y)
    
    # Remove batch dimension
    if hasattr(model, 'sequential') and model.sequential:
        y = y.squeeze(dim=1)
        yhat = yhat.squeeze(dim=1)
    # Convert y and yhat to vector
    y = y[:, 0].detach()
    yhat = yhat[:, 0].detach()
    return loss.item(), y, yhat


def calculateLoss(yhat, y):
    dy = y[1:] - y[:-1]
    dyhat = yhat[1:] - yhat[:-1]
    err = (dy - dyhat)
    loss = th.pow(err, 2).mean()
    return loss.sum()


def savePlot(plot_file_name, df: pd.DataFrame, title: str):
    if (len(df) < 2):
        return
    try:
        # Draw and save chart
        plt.ioff()
        fig = plt.figure(figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        df.plot(kind='line', ax=ax, sharex=True, sharey=True, title=title, grid=True, logy=True)
        fig.tight_layout()
        fig.savefig(plot_file_name)
        plt.close(fig)
    except:
        pass


def denormalize(v, target_min, target_max, target_std, preprocess):
    # Convert output values to prices
    if (preprocess == 'diff'):
        # Denormalize StandardScaler
        v = v * target_std
        # Convert differences to prices
        v = v.cumsum(dim=0)
    elif (preprocess == 'logret'):
        # Denormalize StandardScaler
        v = v * target_std
        # v = v.exp()
        # for i in range(y.size(0)):
        #     v[i] = v[i] * (v[i - 1] if (i > 0) else 1000.0)
        v = v.cumsum(dim=0)
    else:
        # Denormalize MinMaxScaler
        v = v * (target_max - target_min) + target_min
    return v


def saveChart(chart_file_name, y, yhat):
    try:
        fig = plt.figure(figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(y.numpy(), label='y')
        ax.plot(yhat.numpy(), label='yhat')
        ax.legend()
        fig.tight_layout()
        fig.savefig(chart_file_name)
        plt.close(fig)
    except:
        pass


def weightDecay(param_groups, weight_decay):
    if (weight_decay is None) or (weight_decay <= 0) or (weight_decay >= 1):
        return
    for group in param_groups:
        for param in group['params']:
            param.data = param.data.add(-weight_decay * group['lr'], param.data)


def calculate_fan_in_and_fan_out(tensor):
    dim = tensor.dim()
    if (dim < 2):
        raise ValueError('Fan in and fan out can not be computed for tensor with fewer than 2 dimensions')
    if (dim == 2): # Linear
        fan_out = tensor.size(0)
        fan_in = tensor.size(1)
    else:
        num_output_fmaps = tensor.size(0)
        num_input_fmaps = tensor.size(1)
        receptive_field_size = 1
        if (dim > 2):
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def checkWeights(param_groups):
    for group in param_groups:
        for param in group['params']:
            if isinstance(param, th.Tensor):
                if (param.data.dim() >= 2):
                    fan_in, fan_out = calculate_fan_in_and_fan_out(param.data)
                    # bound = 1 / math.sqrt(fan_in)
                    bound = math.sqrt(3*2 / (fan_in + fan_out))
                    param.data[(param.data != param.data) + (param.data == 0)] = th.Tensor(1).uniform_(-bound, bound)
                else:
                    param.data[(param.data != param.data)] = 0
                param.data.clamp_(min=-1e+10, max=1e+10)


if __name__ == '__main__':
    main()
    