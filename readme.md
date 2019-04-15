<p align="left">
    <a href="https://www.python.org/">
        <img src="https://ForTheBadge.com/images/badges/made-with-python.svg"
            alt="python"></a> &nbsp;
    <br />
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square"
            alt="MIT license"></a> &nbsp;
</p>

# Index project
This project initially aimed to reproduce result of this article by Sebastian Heinz:
[A simple deep learning model for stock price prediction using TensorFlow](https://medium.com/mlreview/a-simple-deep-learning-model-for-stock-price-prediction-using-tensorflow-30505541d877)

His TensorFlow project is available online at: https://github.com/sebastianheinz/stockprediction

Briefly: his network consists of 5 sequential dense layers.
It takes 500 stock prices as input and outputs index price prediction for the next 1 minute.
Input and output prices are normalized with **MinMaxScaler** from **sklearn.preprocessing** package.

> **I managed to reproduce his results.
> After that I performed some analysis and came to conclusion
> that this scheme can not predict price movements with enough accuracy.**

I came a little further and tried different input values and trained model for different outputs.
Without significant success, though.

It is not surprise that stock market data is noisy especially at short time intervals.
And it is so volatile that it leaves merely no hope for neural networks to converge.

Below you may find detailed description of data processing, model training and testing.

## Data
I downloaded data from original article from [here](http://files.statworx.com/sp500.zip) 
It contains prices for the period from April to July 2017
including 500 stocks and S&P500 index with 1 minute interval.

You may find CSV data file at: `SP500/sp500_prices.csv.gz`

Note: I shifted S&P500 index price 1 minute to the future, because in the original work
they were shifted 1 minute back to the past.
But my script preprocesses data on-the-fly depending on the argument `target_offset`
which specifies the number of periods to shift target price column.
This allows you to try and test different target prediction offsets.
But it requires target prices to be initially in their true time, 'unshifted'.

Generally speaking you can choose to feed into neural network one of the following:
- Prices. Of course, you should normalize prices. I suggest the best way is **MinMaxScaler**,
  as we don't have any outliers in stock prices.
- Differences of prices: *(Latest_Price - Price_One_Interval_Ago)*.
  These values can contain some outliers with high magnitude. 
  To normalize these values you can use: **StandardScaler(with_mean=False, with_std=True)**.
  See note below.
- Logreturns: *Ln(Latest_Price/Price_One_Interval_Ago)*.
  These values are less likely to contain outliers than differences.
  Again **StandardScaler(with_mean=False, with_std=True)** or even **MaxAbsScaler**
  are appropriate to normalize them.

Note: I suggest using **StandardScaler** with argument
`with_mean=False` in order to keep signs of price changes. I.e. if over last
interval price grew up this normalized value should have positive sign,
and negative - if price fell down. I believe this is important for NN to learn smoothly.  

> When speaking about normalization I encourage you to read this article:
> [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler)

> And also comparision of different scaling methods:
> [Compare the effect of different scalers on data with outliers](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py)

Functions to work with dataset are located in `data.py`

### Loading dataset

Loading dataset from CSV file is fairly easy when using **pandas** package:
```python
import pandas as pd

def loadDataset(file_name):
    dataset = pd.read_csv(file_name)
    dataset['DATETIME'] = pd.to_datetime(dataset['DATETIME'])
    dataset.set_index('DATETIME', inplace=True)
    return dataset
```
Note: after reading CSV file we convert _DATETIME_ column into `dtype=datetime64` type.
And set this column as index for _dataset_.

### Splitting dataset into train and test set
Since we are working with time-series we should be careful when splitting dataset.
It would be wrong to randomly assing each interval sample either to train or to test datasets.
Example of such **wrong split** looks like this: 
![Incorrect way to split dataset](img/train_test_incorrect_split.png)

In this case model will easily learn the curve and won't generalize any latent knowledge behind.

The best way would be to split the whole time-series into two unequal parts,
like author of the original work did.

But I prefer another approach: choose some big interval like week, month or quarter,
and split each big interval of source sequence into two unequal parts: train and test.
Result of such **correct split** would look like this:
 
![Correct way to split dataset](img/train_test_correct_split.png)

Here each month is split into train series (blue) and test series (red).

### Preprocessing data

Preprocessing is implemented in function `preprocessDataset` in `data.py`.

We are given the following arguments:
- `dataset` as **pandas.DataFrame**, where target price is the last column.
- `target_offset` - the number of periods to make prediction into the future.
   I.e. with `target_offset=1` we train model to predict target price in 1 minute.

If you choose to feed into model simply _prices_ then you need just to shift target price for `target_offset` intervals:
```python
    input_columns = dataset.columns[:-1]
    target_columns = dataset.columns[-1:]

    dataset[target_columns] = dataset[target_columns].shift(periods=-target_offset)
```

Computing _differences_ of prices is very easy with pandas:
```python
    # Converting absolute input price values into differences
    dataset[input_columns] = dataset[input_columns].diff(periods=1)
    # Converting absolute target price into differences with target_offset
    dataset[target_columns] = dataset[target_columns].diff(periods=target_offset)
    dataset[target_columns] = dataset[target_columns].shift(periods=-target_offset)
```

Computing _logreturns_ is just a little bit trickier:
```python
    # Converting input prices into log-returns
    dataset[input_columns] = np.log(1 + dataset[input_columns].pct_change(periods=1))
    # Converting target prices into log-returns
    dataset[target_columns] = np.log(1 + dataset[target_columns].pct_change(periods=target_offset))
    dataset[target_columns] = dataset[target_columns].shift(periods=-target_offset)
```

### Normalizing data

Dataset normalization is essential to speed-up model training.

You can use **sklearn.preprocessing** package to normalize **pandas.DataFrame** dataset.

I initialize `scaler` in `train.py` basing on `preprocess` argument as follows:
```python
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # Setup appropriate scaler depending on preprocess argument
    scaler = StandardScaler(with_mean=False, with_std=True) if (preprocess == 'diff') or (preprocess == 'logret')\
        else MinMaxScaler()
```

This scaler is then used in `normalizeDataset` function in `data.py`:
```python
    dataset[dataset.columns] = scaler.fit_transform(dataset[dataset.columns])
```

That is it! Simple and quick solution, thanks to **sklearn.*** packages!

## Model

Model from original article had the following structure:

 N | Layer | Input Size | Output Size
--- | ----- | ----- | ----
 1 | Linear | 500 | 1024
 2 | Linear | 1024 | 512
 3 | Linear | 512 | 256
 4 | Linear | 256 | 128
 5 | Linear | 128 | 1

In this project you may find it in `modules/fc5.py`

There you may find some other models that I have tried:
- `winfc5.py` - five dense layers that work in 'windowed' mode.
  Model takes as input 10 last intervals with all stock prices for each interval.
  I.e. for S&P500 it receives 10x500 = 5000 numbers as input.
- `lstm1.py` - recurrent network with one output dense layer.
  Models takes prices for current interval and keeps

## Training

In order to train a model you need first to specify parameters for `train.py` either in `Params` dictionary
or in a command line:
```python
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
```



## Testing


## Results

Then I constructed a model on PyTorch 