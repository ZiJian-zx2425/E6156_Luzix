from flask import Flask
from sklearn.metrics import roc_auc_score
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.regularizers import l1
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.regularizers import l1
from keras.layers import Bidirectional, GRU
from keras.callbacks import ModelCheckpoint
import keras

_TEST_RATIO = 0.2
_VALIDATION_RATIO = 0.1

def load_data_simple(seqFile, labelFile, timeFile=''):
    with open(seqFile, 'rb') as f:
        sequences = pickle.load(f)
    with open(labelFile, 'rb') as f:
        labels = np.array(pickle.load(f))
    times = None
    if len(timeFile) > 0:
        with open(timeFile, 'rb') as f:
            times = pickle.load(f)

    dataSize = len(labels)
    ind = np.random.permutation(dataSize)
    nTest = int(_TEST_RATIO * dataSize)
    nValid = int(_VALIDATION_RATIO * dataSize)

    test_indices = ind[:nTest]
    valid_indices = ind[nTest:nTest + nValid]
    train_indices = ind[nTest + nValid:]

    train_set_x = [sequences[i] for i in train_indices]
    train_set_y = labels[train_indices]
    test_set_x = [sequences[i] for i in test_indices]
    test_set_y = labels[test_indices]
    valid_set_x = [sequences[i] for i in valid_indices]
    valid_set_y = labels[valid_indices]

    train_set_t = test_set_t = valid_set_t = None

    if times is not None:
        train_set_t = [times[i] for i in train_indices]
        test_set_t = [times[i] for i in test_indices]
        valid_set_t = [times[i] for i in valid_indices]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]

    if times is not None:
        train_set_t = [train_set_t[i] for i in train_sorted_index]
        valid_set_t = [valid_set_t[i] for i in valid_sorted_index]
        test_set_t = [test_set_t[i] for i in test_sorted_index]

    train_set = (train_set_x, train_set_y, train_set_t)
    valid_set = (valid_set_x, valid_set_y, valid_set_t)
    test_set = (test_set_x, test_set_y, test_set_t)

    return train_set, valid_set, test_set


train_set, valid_set, test_set = load_data_simple('outfile.3digitICD9.seqs', 'outfile.morts', 'outfile.time_seqs')

train_set_x, train_set_y, train_set_t = train_set
valid_set_x, valid_set_y, valid_set_t = valid_set
test_set_x, test_set_y, test_set_t = test_set

def padMatrixWithTime(seqs, times, inputDimSize, useLogTime):
    lengths = np.array([len(seq) for seq in seqs]).astype('int32')
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples, inputDimSize)).astype('float32')
    for idx, seq in enumerate(seqs):
        for i, subseq in enumerate(seq):
            x[i, idx, subseq] = 1.

    t = np.zeros((maxlen, n_samples)).astype('float32')
    for idx, time in enumerate(times):
        t[:lengths[idx], idx] = time

    if useLogTime:
        t = np.log(t + 1.)

    return x, t, lengths

train_set_x, train_set_t, train_lengths = padMatrixWithTime(train_set_x, train_set_t, inputDimSize=942, useLogTime=True)
valid_set_x, valid_set_t, valid_lengths = padMatrixWithTime(valid_set_x, valid_set_t, inputDimSize=942, useLogTime=True)
test_set_x, test_set_t, test_lengths = padMatrixWithTime(test_set_x, test_set_t, inputDimSize=942, useLogTime=True)

number_of_ones = test_set_y.count(1)
number_of_zeros = test_set_y.count(0)
ratio = number_of_ones / (number_of_zeros + number_of_ones)
print(number_of_ones)
print(number_of_zeros)
print(ratio)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def prepare_data_for_logreg(x_set, y_set, lengths):
    num_samples = len(y_set)
    prepared_x_set = []
    for i in range(num_samples):
        # For each sample, we use its sequence length to get the specific consultation records
        x = x_set[:lengths[i], i, :]

        # We use the mean of each sample as the feature for logistic regression
        x = np.mean(x, axis=0)

        prepared_x_set.append(x)
    return np.array(prepared_x_set), np.array(y_set)


# Prepare the data
train_set_x, train_set_y = prepare_data_for_logreg(train_set_x, train_set_y, train_lengths)
valid_set_x, valid_set_y = prepare_data_for_logreg(valid_set_x, valid_set_y, valid_lengths)
test_set_x, test_set_y = prepare_data_for_logreg(test_set_x, test_set_y, test_lengths)

import joblib

best_auc = 0.0  # Initialize the best AUC to a low value
best_model = None  # Placeholder for the best model

for epoch in range(10):
    # Train the logistic regression model
    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(train_set_x, train_set_y)

    # Compute the AUC for validation set
    valid_pred_y = logreg.predict_proba(valid_set_x)[:, 1]
    valid_auc = roc_auc_score(valid_set_y, valid_pred_y)
    print(f'Epoch {epoch + 1}, Validation AUC: {valid_auc}')

    # If this model is better than previous ones, save it
    if valid_auc > best_auc:
        best_auc = valid_auc
        best_model = logreg
        # Save the model to a file
        joblib.dump(best_model, 'best_logreg_model.pkl')

# After all epochs, the saved model is the best one
print(f'Best Validation AUC over 10 epochs: {best_auc}')
