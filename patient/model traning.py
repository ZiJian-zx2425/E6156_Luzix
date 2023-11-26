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
    np.random.seed(0)
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


train_set_x, train_set_y, train_set_t = train_set
valid_set_x, valid_set_y, valid_set_t = valid_set
test_set_x, test_set_y, test_set_t = test_set

train_set_x, train_set_t, train_lengths = padMatrixWithTime(train_set_x, train_set_t, inputDimSize=942, useLogTime=True)
valid_set_x, valid_set_t, valid_lengths = padMatrixWithTime(valid_set_x, valid_set_t, inputDimSize=942, useLogTime=True)
test_set_x, test_set_t, test_lengths = padMatrixWithTime(test_set_x, test_set_t, inputDimSize=942, useLogTime=True)

from keras.layers import GlobalAveragePooling1D


# This is an input data generator
def data_generator(x_set, t_set, y_set, lengths):
    num_samples = len(y_set)
    for i in range(num_samples):
        # For each sample (or patient), we use its sequence length to get the specific consultation records and timestamps
        x = x_set[:lengths[i], i, :]
        t = t_set[:lengths[i], i, np.newaxis]  # add a new dimension to facilitate concatenation in the next step
        y = y_set[i]

        # Combine x and t
        x = np.concatenate((x, t), axis=-1)  # concatenate along the last axis

        # This generator will produce the consultation data of a patient each time
        yield x, y


from keras import backend as K
from keras.layers import Layer


class FixedAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(FixedAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='normal')
        super(FixedAttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W))
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def create_rnn_model(input_dim, hidden_dim, output_dim):
    model = Sequential()

    # Add a Bidirectional GRU layer with return sequences to keep the sequence structure for attention
    model.add(Bidirectional(GRU(hidden_dim, return_sequences=True), input_shape=(None, input_dim)))

    # Add the Attention Layer
    model.add(FixedAttentionLayer())

    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='sigmoid', kernel_regularizer=l1(0.01)))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=['accuracy'])
    return model


from keras.layers import LSTM

def create_lstm_model(input_dim, hidden_dim, output_dim):
    model = Sequential()

    # Add a Bidirectional LSTM layer with return sequences to keep the sequence structure for attention
    model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True), input_shape=(None, input_dim)))

    # Add the Attention Layer
    model.add(FixedAttentionLayer())

    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='sigmoid', kernel_regularizer=l1(0.01)))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=['accuracy'])
    return model


# Define the input dimension (e.g., the size of the patient records after concatenation with the timestamp)
input_dim = train_set_x.shape[-1] + 1
hidden_dim = 64
output_dim = 1

# Create the model
model = create_rnn_model(input_dim, hidden_dim, output_dim)

num_epochs = 2

max_test_aucs = []
NUM_RUNS = 1  # 例如，您可以设置为5次运行，您可以根据需要更改这个数字

# GRU模型训练和评估
print("Training and evaluating the GRU model...")
gru_model = create_rnn_model(input_dim, hidden_dim, output_dim)

for epoch in range(num_epochs):
    print("Epoch %d:" % (epoch + 1))

    # Training for GRU
    train_pred_y = []
    for i, (x, y) in enumerate(data_generator(train_set_x, train_set_t, train_set_y, train_lengths)):
        x = np.reshape(x, (1,) + x.shape)
        y = np.array([y])
        gru_model.train_on_batch(x, y)
        train_pred_y.append(gru_model.predict_on_batch(x)[0][0])
    train_auc = roc_auc_score(train_set_y, train_pred_y)
    print('Training AUC (GRU): ', train_auc)

    # Validation for GRU
    total_val_loss = 0.0
    num_val_samples = 0
    valid_pred_y = []
    for i, (x, y) in enumerate(data_generator(valid_set_x, valid_set_t, valid_set_y, valid_lengths)):
        x = np.reshape(x, (1,) + x.shape)
        y = np.array([y])
        val_loss = gru_model.test_on_batch(x, y)[0]
        total_val_loss += val_loss
        num_val_samples += 1
        valid_pred_y.append(gru_model.predict_on_batch(x)[0][0])
    avg_val_loss = total_val_loss / num_val_samples
    print("Validation Loss (GRU): %f" % avg_val_loss)
    valid_auc = roc_auc_score(valid_set_y, valid_pred_y)
    print('Validation AUC (GRU): ', valid_auc)

    # Checkpointing
    if epoch == 0 or valid_auc > max_valid_auc:
        gru_model.save('best_gru_model.keras')
        max_valid_auc = valid_auc

# LSTM模型训练和评估
print("\nTraining and evaluating the LSTM model...")
lstm_model = create_lstm_model(input_dim, hidden_dim, output_dim)

for epoch in range(num_epochs):
    print("Epoch %d:" % (epoch + 1))

    # Training for LSTM
    train_pred_y = []
    for i, (x, y) in enumerate(data_generator(train_set_x, train_set_t, train_set_y, train_lengths)):
        x = np.reshape(x, (1,) + x.shape)
        y = np.array([y])
        lstm_model.train_on_batch(x, y)
        train_pred_y.append(lstm_model.predict_on_batch(x)[0][0])
    train_auc = roc_auc_score(train_set_y, train_pred_y)
    print('Training AUC (LSTM): ', train_auc)

    # Validation for LSTM
    total_val_loss = 0.0
    num_val_samples = 0
    valid_pred_y = []
    for i, (x, y) in enumerate(data_generator(valid_set_x, valid_set_t, valid_set_y, valid_lengths)):
        x = np.reshape(x, (1,) + x.shape)
        y = np.array([y])
        val_loss = lstm_model.test_on_batch(x, y)[0]
        total_val_loss += val_loss
        num_val_samples += 1
        valid_pred_y.append(lstm_model.predict_on_batch(x)[0][0])
    avg_val_loss = total_val_loss / num_val_samples
    print("Validation Loss (LSTM): %f" % avg_val_loss)
    valid_auc = roc_auc_score(valid_set_y, valid_pred_y)
    print('Validation AUC (LSTM): ', valid_auc)

    # Checkpointing
    if epoch == 0 or valid_auc > max_valid_auc:
        lstm_model.save('best_lstm_model.keras')
        max_valid_auc = valid_auc