# RNN time-series prediction on sine-waves and TSLA stock prices

Given an discete scalar sequence *f(t)* of your choice (i.e. stock prices, audio signals, population growth, etc), we can attempt to learn local patterns of the sequence by training a recurrent neural network (RNN) on randomly sampled segments of the sequence. The great utility of this method after training is that we can then predict the next time-step value of *f(t)* given any segment. Furthermore, recursively applying its own predictions back into itself leads to a time-series extrapolation of *f(t)* to infinity. In this example, we train a vanilla RNN to make such time-series predictions (black = data, red = prediction) on:
 
 
* (1) A truncated sine-wave of constant frequency  (set `RNN_data_mode = 'stocks'`)
 
<img src="https://github.com/ankonzoid/Deep-NN-Python-Tutorials/blob/master/RNNs/RNN_timeseries/plots/timeseries_sinewave.png" width="50%" align="center">

The vanilla RNN model performs fairly well in extrapolating the periodicity of the sine wave, though much of this success can be attributed to the smoothness and simplicity of the sine wave.

 
* (2) Closing prices of TSLA stock between Jan 1st, 2014 and December 31st, 2016 (set `RNN_data_mode = 'stocks'`)

<img src="https://github.com/ankonzoid/Deep-NN-Python-Tutorials/blob/master/RNNs/RNN_timeseries/plots/timeseries_stocks.png" width="50%" align="center">

There is a slight hope that the vanilla RNN model makes a somewhat realistic prediction for the TSLA daily closing prices, which makes it somewhat of a helper with determining the short-term price momentum. What is omitted in this graph though is that the long-term behaviour is too simple and unrealistic which tends to be a very ubiquitous problem with RNNs (in our example run, the price shoots up to unrealistic prices and continues in a straight line). This sort of behaviour disqualifies this vanilla RNN model from being a solid predictor of stock prices in the long-run (as one might have expected and been critical of). 


### Usage:

To perform time-series predictions on the truncated sine-wave using the pre-existing trained models, run the command

> python RNN_timeseries.py

The adjustable parameters in `RNN_timeseries.py`:

* To use pre-existing trained models set `train_model = False`, and to train the model from scratch set `train_model = True`

* To implement the sine-wave time-series set `RNN_data_mode = 'sinewave'`, and to implement the TSLA stock price time-series set `NN_data_mode = 'stocks'`


### Libraries required: 

* tensorflow, numpy
* pandas_datareader (for TSLA stock price retrieval)