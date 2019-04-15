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
  Again **StandardScaler(with_mean=False, with_std=True)** or even **MaxAbsScaler**.
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

##### Loading dataset

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

##### Splitting dataset into train and test set
Since we are working with time-series we should be careful when splitting dataset.
It would be wrong to randomly assing each interval sample either to train or to test datasets.
Example of such **wrong split** looks like this: 
![Incorrect way to split dataset](img/train_test_incorrect_split.png)

In this case model will easily learn the curve and won't generalize any latent knowledge behind.

The best way would be to split the whole time-series into two unequal parts,
like author of the original work did.

But I prefer another approach: choose some big interval like week, month or quarter,
and for split each big interval of source sequence into two unequal parts: train and test.
Result of such **correct split** would look like this:
 
![Correct way to split dataset](img/train_test_correct_split.png)

Here each month is split into train series (blue) and test series (red).

## Model

Model from original article had the following structure:

 N | Layer | Input Size | Output Size
--- | ----- | ----- | ----
 - | Input | 500 | 
 1 | Linear | 500 | 1024
 2 | Linear | 1024 | 512
 3 | Linear | 512 | 256
 4 | Linear | 256 | 128
 5 | Linear | 128 | 1

In this project you may find it in `modules/fc5.py`


## Training


## Testing


## Results

Then I constructed a model on PyTorch 