"""

 rnn_timeseries.py (author: Anson Wong / git: ankonzoid)

 Basic RNN time-series prediction by training on multiple random subsequences of a larger sequence of 1D data. We force the unsupervised problem to be supervised by inputting subsequences, and their shifted to the future subsequences as labels.

 Example training output:
     ...
     ...
 	[epoch 94/100] MSE: 304.7425842285156
	[epoch 95/100] MSE: 301.2587585449219
	[epoch 96/100] MSE: 298.9361877441406
	[epoch 97/100] MSE: 298.0180969238281
	[epoch 98/100] MSE: 296.6414794921875
	[epoch 99/100] MSE: 294.1679992675781

"""
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import tensorflow as tf

def main():
    style.use('ggplot')

    # ===================================
    # Set parameters
    # ===================================
    train_model = True
    RNN_data_mode = 'stocks'  # 'sinewave' or 'stocks'

    # ===================================
    # Set data
    # ===================================
    if RNN_data_mode == 'sinewave':

        # Create sine wave data
        n_periods = 2  # the frequency of the signal
        n_points_per_period = 100
        n_points = int(n_periods*n_points_per_period)  # period

        X_data_full = np.array([np.sin(2 * np.pi * n_periods * (i / n_points)) for i in np.arange(n_points)])

        # Set up NN arch parameters
        n_steps = 100  # number of time-series steps per batch
        n_neurons = 100  # neurons in hidden layer

        # Set up training sampling parameters
        n_epochs = 300  # number of epochs
        random_train = True  # use
        n_train = 5000  # number of training examples

        # Time-series prediction parameters
        n_steps_extrapolate = 500

    elif RNN_data_mode == 'stocks':

        # Download stock data
        import datetime as dt
        import pandas_datareader.data as web

        ticker = "TSLA"
        start = dt.datetime(2014, 1, 1)
        end = dt.datetime(2016, 12, 31)

        print("Attempting to collect '{0}' historical daily charts from '{1}' to '{2}'".format(ticker, start, end))
        df = web.DataReader(ticker, 'google', start, end)
        print(df.head())

        X_data_full = df.values[:, 3]  # 0: open, 1: high, 2: low, 3: close, 4: volume (collect daily closing prices)

        # Set up NN arch parameters
        n_steps = 100  # number of time-series steps per batch
        n_neurons = 200  # neurons in hidden layer

        # Set up training sampling parameters
        n_epochs = 100  # number of epochs
        random_train = True  # use
        n_train = 5000  # number of training examples

        # Time-series prediction parameters
        n_steps_extrapolate = 10

    else:
        raise Exception("Invalid RNN data mode!")

    # ===================================
    # Model filename
    # ===================================
    model_filename = 'models_' + RNN_data_mode + '/RNN_timeseries_' + RNN_data_mode

    if 0:
        plt.figure(1)
        plt.plot(X_data_full, '-')
        plt.show()

    # Set X_data and y_data from X_data_full (by shifting one time step)
    X_data = X_data_full[:-1]  # input sequences: remove the last element from X_train_full
    y_data = X_data_full[1:]  # target sequence: remove the first element from X_train_full (1 day shifted forward)

    # Print data shapes
    print("X_data.shape = {0}".format(X_data.shape))
    print("y_data.shape = {0}".format(y_data.shape))

    # ============================
    # Set parameters
    # ============================
    n_inputs = 1  # number of input
    n_outputs = 1  # number of output
    overfit = False  # train on one sequence (overfit) or random set of sequences?

    # ============================
    # Build RNN architecture
    # ============================
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
    cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
        output_size=n_outputs)  # wrapping the cell by adding FC layer of linear neurons
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    learning_rate = 0.01
    loss = tf.reduce_mean(tf.square(outputs - y))  # set L2 loss

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Adam optimizer
    training_op = optimizer.minimize(loss)  # minimize loss function
    init = tf.global_variables_initializer()

    # ============================
    # Start tensorflow session
    # ============================

    # Set saver for model
    saver = tf.train.Saver()  # create instance of saver class

    with tf.Session() as sess:

        if not train_model:
            # ============================
            # Load model
            # ============================
            print("Loading model from file: %s" % model_filename)
            saver.restore(sess, model_filename)

        else:
            # ============================
            # Train model
            # ============================
            if overfit:
                idx_start = 0
                X_train, y_train = get_batch_seq(X_data, y_data, n_steps, n_inputs, n_outputs, idx_start)
            else:
                X_train, y_train = get_random_batch_seq(X_data, y_data, n_steps, n_inputs, n_outputs, n_train)

            print("X_train.shape = {0}".format(X_train.shape))
            print("y_train.shape = {0}".format(y_train.shape))

            init.run()
            for i in range(n_epochs):

                sess.run(training_op, feed_dict={X: X_train, y: y_train})  # train

                if i % int(n_epochs/100) == 0:
                    mse = loss.eval(feed_dict={X: X_train, y: y_train})
                    print("\t[epoch {0}/{1}] MSE: {2}".format(i, n_epochs, mse))

            # ===============================
            # Save model weights to disk
            #
            # weights = .data-00000-of-00001
            # .index
            # NN arch = .meta
            # checkpoint
            #
            # ===============================
            print("Saving model to file: %s" % model_filename)
            saver.save(sess, model_filename)  # save session

        # ============================
        # Test model
        # ============================
        if overfit:
            idx_start = 0
            batch_size_test = 1
            X_test, y_test = get_batch_seq(X_data, y_data, n_steps, n_inputs, n_outputs, idx_start)
        else:
            batch_size_test = 1
            X_test, y_test = get_random_batch_seq(X_data, y_data, n_steps, n_inputs, n_outputs, batch_size_test)

        y_pred = sess.run(outputs, feed_dict={X: X_test})

        err_sum = 0
        for i in range(batch_size_test):
            err = np.sum(np.abs(y_pred[i] - y_test[i])) / len(y_test[i])
            err_sum += err
            print("[batch {0}] err = {1}".format(i+1, err))
            if 0:
                plt.figure(1)
                p_xtest, = plt.plot(range(len(X_test[i])), X_test[i], 'k.')
                p_ytest, = plt.plot(range(len(y_test[i])), y_test[i], 'bo')
                p_ypred, = plt.plot(range(len(y_pred[i])), y_pred[i], 'ro')
                plt.legend([p_xtest, p_ytest, p_ypred], ["xtest", "ytest", "ypred"], loc=1)
                plt.show()
        err_avg = err_sum / batch_size_test
        print(" -> err (batch avg) = {0}".format(err_avg))

        # ============================
        # Make extrapolation predictions
        # ============================
        if random_train:
            idx_start = 0
            X_extrapolate, y_extrapolate = get_batch_seq(X_data, y_data,
                                                         n_steps, n_inputs, n_outputs,
                                                         idx_start)
        else:
            X_extrapolate, y_extrapolate = get_random_batch_seq(X_data, y_data,
                                                                n_steps, n_inputs, n_outputs,
                                                                n_train)

        X_extrapolate_append = X_extrapolate.copy()
        n_extrapolate_append = n_steps
        for i in range(n_steps_extrapolate):
            print("Extrapolating {0}/{1}...".format(i+1, n_steps_extrapolate))

            y_pred_extrapolate = sess.run(outputs, feed_dict={X: X_extrapolate})

            # Concatenate to original sequence for plotting purposes
            X_extrapolate_append = np.append(X_extrapolate_append[:,0:,:], y_pred_extrapolate[:,-1,:])
            n_extrapolate_append += 1
            X_extrapolate_append = X_extrapolate_append.reshape(-1, n_extrapolate_append, 1)

            # Create new n_step sequence for next input
            X_extrapolate = np.append(X_extrapolate[:,1:,:], y_pred_extrapolate[:,-1,:])
            X_extrapolate = X_extrapolate.reshape(-1, n_steps, 1)

        # Plot extrapolation predictions
        if 1:
            plt.figure(1)
            X_plot = X_extrapolate_append.flatten()

            x_plot_truth = np.arange(0, n_steps)
            y_plot_truth = X_plot[:n_steps]
            x_plot_extrapolate = np.arange(n_steps, len(X_plot))
            y_plot_extrapolate = X_plot[n_steps:]

            plt_xtruth, = plt.plot(x_plot_truth, y_plot_truth, 'ko')  # plot truth black
            plt_xextrapolate, = plt.plot(x_plot_extrapolate, y_plot_extrapolate, 'ro')  # plot pred red
            plt.legend([plt_xtruth, plt_xextrapolate], ["truth", "prediction"], loc=1)

            plt.savefig("plots/timeseries_%s.png" % RNN_data_mode,
                        bbox_inches='tight')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Side functions
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# get_random_batch_seq:
def get_random_batch_seq(X_data, y_data, n_steps, n_inputs, n_outputs, batch_size):
    X_batch = []
    y_batch = []
    n_train = len(X_data)
    for i in range(batch_size):
        idx = random.randint(0, n_train - n_steps)
        X_batch.append(X_data[idx:idx + n_steps])
        y_batch.append(y_data[idx:idx + n_steps])
    X_batch = np.array(X_batch).reshape((-1, n_steps, n_inputs))
    y_batch = np.array(y_batch).reshape((-1, n_steps, n_outputs))
    return X_batch, y_batch

# get_batch_seq:
def get_batch_seq(X_data, y_data, n_steps, n_inputs, n_outputs, idx_start):
    X_batch = []
    y_batch = []
    idx = idx_start
    X_batch.append(X_data[idx:idx + n_steps])
    y_batch.append(y_data[idx:idx + n_steps])
    X_batch = np.array(X_batch).reshape((-1, n_steps, n_inputs))
    y_batch = np.array(y_batch).reshape((-1, n_steps, n_outputs))
    return X_batch, y_batch

if __name__ == '__main__':
    main()
