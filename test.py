import os
import torch as th
import torch.nn as nn
import argparse
import matplotlib
#matplotlib.use('Agg')
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
    weight_decay=300,
    early_stop=0.3
)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Index')
    parser.add_argument('config', type=str, default='RTSI',
        help='Name of index subdirectory which contains configuration and data to process.')
    parser.add_argument('-model', type=str, default=Params['model'],
        help='Name of model to train. Choose one from models subdirectory.')
    parser.add_argument('-preprocess', type=str, default=Params['preprocess'],
        help='Choose how to preprocess prices: none, diff or logret, default: diff')
    parser.add_argument('-target-offset', type=int, default=Params['target_offset'],
        help='How many future intervals to predict, int, default: 1')
    args = parser.parse_args()
    preprocess = args.preprocess
    scaler = 'MaxAbsScaler' if (preprocess == 'diff') or (preprocess == 'logret') else 'MinMaxScaler'
    target_offset = args.target_offset

    # Setup torch
    th.set_num_threads(2)
    th.set_default_dtype(th.float32)
    th.set_default_tensor_type(th.FloatTensor)

    # Load configuration
    Config = __import__(args.config, globals(), locals())
    assert hasattr(Config, 'Dataset_File')
    assert hasattr(Config, 'Input_Size')

    # Initialize model
    print('Initializing model')
    _module = __import__('models.' + args.model, globals(), locals(), fromlist=['Model'], level=0)
    Model = _module.Model
    model = Model(input_size=Config.Input_Size, output_size=1, hidden_size=1024)

    # Construct file names
    model_file_name = args.config + '/' + args.model + '.pt'
    plot_file_name = args.config + '/' + args.model + '.png'
    if (not os.path.isfile(model_file_name)):
        raise FileNotFoundError(model_file_name)

    # Load model parameters
    state_dict = th.load(model_file_name)
    model.load_state_dict(state_dict, strict=True)
    print(model, '\n')

    # Load dataset
    print('Loading dataset from {}'.format(Config.Dataset_File))
    dataset = data.loadDataset(Config.Dataset_File)
    print(dataset.head(10), '\n')

    # print('Splitting dataset into train and test')
    # train, test = data.splitDataset(dataset)
    # print(' Train: total {} records'.format(len(train)))
    # print(' Test: total {} records'.format(len(test)))
    # dataset = test
    
    # Preprocess dataset
    print('Preprocessing dataset, method={}, target_offset={}'.format(preprocess, target_offset))
    data.preprocessDataset(dataset, method=preprocess, target_offset=target_offset)
    print(dataset.head(10), '\n')

    # Save min, max of target values
    target = dataset.iloc[:, -1]
    target_min, target_max, target_mean, target_std = target.min(), target.max(), target.mean(), target.std()
    del target
    print('Target min={} max={} mean={} std={}\n'.format(target_min, target_max, target_mean, target_std))

    print('Normalizing dataset...')
    data.normalizeDataset(dataset, scaler=scaler)
    print(dataset.head(10), '\n')

    # Read input and target
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1:].values
    x = th.tensor(x, dtype=th.get_default_dtype())
    y = th.tensor(y, dtype=th.get_default_dtype())
    # Add batch dimension if needed
    if hasattr(model, 'sequential') and model.sequential:
        x = x.unsqueeze(dim=1)
        y = y.unsqueeze(dim=1)

    # Pass whole sequence through model. Get prediction
    model.eval()
    model.reset()
    yhat = model.forward(x)

    # Cut down y to be the same length as yhat as some windowed models may return a shorter sequence
    if (y.size(0) > yhat.size(0)):
        y = y[-yhat.size(0):]

    # Calculate loss
    criterion = nn.MSELoss(reduction='mean')
    loss = criterion(yhat, y)
    loss = loss.sum()

    # Remove batch dimension
    if hasattr(model, 'sequential') and model.sequential:
        y = y.squeeze(dim=1)
        yhat = yhat.squeeze(dim=1)

    # Convert y and yhat to vector
    y = y[:, 0].detach()
    yhat = yhat[:, 0].detach()
    
    # Total count of predictions made
    n = yhat.size(0)
    
    # Calculate prediction error
    if (preprocess == 'diff'):
        # Denormalize StandardScaler
        y = y * target_std
        yhat = yhat * target_std
        # Calculate err
        d = y * yhat
        err = (y - yhat)
        # Convert differences to prices
        y = y.cumsum(dim=0)
        yhat = yhat.cumsum(dim=0)
    elif (preprocess == 'logret'):
        # Denormalize StandardScaler
        y = y * target_std
        yhat = yhat * target_std
        # Calculate err
        d = y * yhat
        err = (y - yhat)
        # Convert log(returns) to prices
        y = y.cumsum(dim=0)
        yhat = yhat.cumsum(dim=0)
    else:
        # Denormalize MinMaxScaler
        y = y * (target_max - target_min) + target_min
        yhat = yhat * (target_max - target_min) + target_min
        # Calculate err
        dy = y[1:] - y[:-1]
        dyhat = yhat[1:] - yhat[:-1]
        d = dy * dyhat
        err = (dy - dyhat)

    # Number of times NN correctly predicted future price movement up or down
    n_pos = (d >= 0).sum()
    # Number of times NN failed to predict future price movement up or down
    n_neg = (d < 0).sum()
    # Mean error when NN correctly predicted future price movement
    err_pos = th.sqrt(th.pow(err[d >= 0], 2).mean())
    # Mean error when NN failed to predict future price movement
    err_neg = th.sqrt(th.pow(err[d < 0], 2).mean())
    
    # Print out results
    print('Loss: {}'.format(loss.item()))
    print('Number of source timesteps: {}'.format(len(dataset)))
    print('Total number of predictions: {}'.format(n))
    print('Count of correct predictions: {}'.format(n_pos))
    print('Count of incorrect predictions: {}'.format(n_neg))
    print('Mean error for correct prediction: {}'.format(err_pos))
    print('Mean error for incorrect prediction: {}'.format(err_neg))

    # Setup plot
    plt.ioff()
    fig = plt.figure(figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(y.numpy(), label='y')
    ax.plot(yhat.numpy(), label='yhat')
    plt.title('Test')
    ax.legend()
    plt.pause(0.5)
    plt.show()
    