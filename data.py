import os
import math
from sklearn.base import TransformerMixin
import sklearn.preprocessing as preprocessing
import numpy as np
import pandas as pd
import argparse


def loadDataset(file_name):
    database = pd.read_csv(file_name)
    database['DATETIME'] = pd.to_datetime(database['DATETIME'])
    database.set_index('DATETIME', inplace=True)
    return database


def splitDataset(dataset, train_test_ratio=0.7, interval='month'):
    assert (0 < train_test_ratio < 1)
    assert hasattr(dataset.index, interval)
    intervals = getattr(dataset.index, interval)
    train, test = None, None
    interval_start_i = 0
    last_interval = None
    for i in range(len(dataset)):
        if (last_interval is not None) and (intervals[i] != last_interval):
            # Split last period into train and test
            n = (i - interval_start_i)
            if (n > 0):
                n_train = int(math.floor(0.5 + n * train_test_ratio))
                train_subset = dataset.iloc[interval_start_i:(interval_start_i + n_train)]
                test_subset = dataset.iloc[(interval_start_i + n_train):i]
                train = pd.concat([train, train_subset]) if (train is not None) else train_subset
                test = pd.concat([test, test_subset]) if (test is not None) else test_subset
            # Set start for next period
            interval_start_i = i
        # Save last period
        last_interval = intervals[i]

    # Append last interval
    n = (len(dataset) - interval_start_i)
    if (n > 0):
        n_train = int(math.floor(0.5 + n * train_test_ratio))
        train_subset = dataset.iloc[interval_start_i:(interval_start_i + n_train)]
        test_subset = dataset.iloc[(interval_start_i + n_train):]
        train = pd.concat([train, train_subset]) if (train is not None) else train_subset
        test = pd.concat([test, test_subset]) if (test is not None) else test_subset
        
    # Return train and test datasets
    train = train.copy()
    test = test.copy()
    return train, test


def preprocessDataset(dataset, target_offset=1, method='diff', target_columns=None):
    if isinstance(target_columns, (list, tuple)):
        input_columns = []
        for column in dataset.columns:
            if (column not in target_columns):
                input_columns.append(column)
    else:
        input_columns = dataset.columns[:-1]
        target_columns = dataset.columns[-1:]
        
    if (method == 'diff'):
        # Converting absolute input price values into differences
        dataset[input_columns] = dataset[input_columns].diff(periods=1)
        # Converting absolute target price into differences with target_offset
        dataset[target_columns] = dataset[target_columns].diff(periods=target_offset)
        dataset[target_columns] = dataset[target_columns].shift(periods=-target_offset)
    elif (method == 'logret'):
        # Converting input prices into log-returns
        dataset[input_columns] = np.log(1 + dataset[input_columns].pct_change(periods=1))
        # Converting target prices into log-returns
        dataset[target_columns] = np.log(1 + dataset[target_columns].pct_change(periods=target_offset))
        dataset[target_columns] = dataset[target_columns].shift(periods=-target_offset)
    else:
        dataset[target_columns] = dataset[target_columns].shift(periods=-target_offset)
    dataset.dropna(how='any', inplace=True)


def normalizeDataset(dataset, scaler='StandardScaler'):
    # Scale data.
    if isinstance(scaler, TransformerMixin):
        pass
    elif isinstance(scaler, str):
        assert hasattr(preprocessing, scaler)
        scaler_class = getattr(preprocessing, scaler)
        scaler = scaler_class()
    else:
        raise ValueError
    dataset.iloc[:, :] = scaler.fit_transform(dataset.to_numpy())


def iterateDataset(dataset, batch_size, sequence_size=None, sequences_iou=0.8, group_by='date'):
    assert isinstance(batch_size, int) and (batch_size > 0)

    if (sequence_size is None) or (sequence_size <= 0):
        # Iterate over dataset in random order
        indexes = np.random.permutation(len(dataset))
        for i in range(0, len(dataset), batch_size):
            if ((len(dataset) - i) < batch_size):
                break
            batch_indexes = indexes[i:(i + batch_size)]
            batch = dataset.iloc[batch_indexes]
            xs = batch.iloc[:, :-1].values
            ys = batch.iloc[:, -1:].values
            yield xs, ys

    else:
        assert isinstance(sequence_size, int) and (sequence_size > 0)
        # Generate sequences array for starting indexes of all possible sequences
        sequences = []
        sequence_step = max(1, int(sequence_size * (1 - sequences_iou)))
        i = 0
        while (i < len(dataset)):
            if (len(dataset) - i < sequence_size):
                break
            if (group_by is not None) and hasattr(dataset.index, group_by):
                group_start = getattr(dataset.index[i], group_by)()
                group_end = getattr(dataset.index[(i + sequence_size - 1)], group_by)()
                if (group_start != group_end):
                    while (i < len(dataset)) and (group_start != group_end):
                        group_start = getattr(dataset.index[i], group_by)()
                        i += 1
                    continue
            # Add current start index to sequences array
            sequences.append(i)
            # Shift i to the start of next sequence
            i += sequence_step

        # Iterate over sequences in random order
        indexes = np.random.permutation(len(sequences))
        for i in range(0, len(sequences), batch_size):
            if ((len(sequences) - i) < batch_size):
                break
            xs, ys = [], []
            batch_indexes = indexes[i:i + batch_size]
            for j in range(batch_size):
                start = batch_indexes[j]
                sequence = dataset.iloc[start:(start + sequence_size)]
                x = sequence.iloc[:, :-1].values
                y = sequence.iloc[:, -1:].values
                xs.append(x)
                ys.append(y)
            xs = np.array(xs, dtype='float32').transpose((1, 0, 2))
            ys = np.array(ys, dtype='float32').transpose((1, 0, 2))
            yield xs, ys
            

def _loadStock(name, tzinfo='Europe/Moscow'):
    # Read csv file in a default export format of finam.ru
    file_name = name + '.txt'
    if not os.path.isfile(file_name):
        file_name += '.gz'
        if not os.path.isfile(file_name):
            raise FileNotFoundError
    data = pd.read_csv(file_name)
    _, name = os.path.split(name)
    
    # Convert <DATE> and <TIME> columns to one DATETIME column
    datetime = data['<DATE>'].map(lambda x: '{:06d}'.format(x)) + data['<TIME>']
    datetime = pd.to_datetime(datetime, format='%y%m%d%H:%M', exact=True)
    if (tzinfo is not None):
        datetime = datetime.dt.tz_localize(tzinfo)
    data['DATETIME'] = datetime
    
    # Rename <CLOSE> -> name.PRICE
    data.rename(columns={'<CLOSE>': name}, inplace=True)
    
    # Remove unused columns and setup index
    data.drop(labels=['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<VOL>'], axis=1, inplace=True)
    data.set_index('DATETIME', inplace=True)
    return data


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Index')
    parser.add_argument('config', type=str, default='RTSI',
        help='Name of Index subdirectory which contains configuration and data to process')
    args = parser.parse_args()
    
    # Load configuration
    Config = __import__(args.config)
    assert hasattr(Config, 'Default_Timezone')
    assert hasattr(Config, 'Dataset_File')

    dataset = None
    if os.path.isfile(Config.Dataset_File):
        print('Dataset was already created in file {}'.format(Config.Dataset_File))
        exit()

    assert hasattr(Config, 'Source')

    print('Creating dataset')
    if isinstance(Config.Source, list):
        for name in Config.Source:
            print('Processing {}'.format(name))
            data = _loadStock(name, tzinfo=Config.Default_Timezone)
            print(' Loaded {} rows'.format(len(data)))
            dataset = pd.merge(dataset, data, how='outer', left_index=True, right_index=True)\
                if (dataset is not None) else data
    elif isinstance(Config.Source, str):
        print('Processing {}'.format(Config.Source))
        dataset = pd.read_csv(Config.Source)
        datetime = pd.to_datetime(dataset['DATETIME'])
        dataset['DATETIME'] = datetime
        dataset.set_index('DATETIME', inplace=True)
    print(dataset.head(10))

    # Target price column should be the last one
    print('Dropping rows with target=NaN')
    inputs = dataset.columns[:-1]
    target = dataset.columns[-1]
    dataset.dropna(how='any', subset=[target], inplace=True)

    print('Filling missing values by LOCF (last observation carried forward)')
    for column in dataset.columns:
        dataset[column].fillna(method='ffill', inplace=True)

    print('Dropping rows with NaN')
    dataset.dropna(how='any', inplace=True)
    print(dataset.head(10))

    if hasattr(Config, 'Work_Date'):
        print('Removing out of work date rows for period {}'.format(Config.Work_Date))
        dataset = dataset.loc[Config.Work_Date[0]:Config.Work_Date[1]]
        print(dataset.head(10))

    if hasattr(Config, 'Work_Time'):
        print('Removing out of work time rows for period {}'.format(Config.Work_Time))
        dataset = dataset.between_time(*Config.Work_Time)
        print(dataset.head(10))

    print('Saving dataset to file: {}'.format(Config.Dataset_File))
    dataset.to_csv(Config.Dataset_File)
