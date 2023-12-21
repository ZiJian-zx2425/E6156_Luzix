import asyncio
import random
import aiohttp
from flask import Flask
from flask import request, redirect, session, url_for, render_template, request, jsonify
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
import keras
from flask_cors import CORS
_TEST_RATIO = 0.2
_VALIDATION_RATIO = 0.1
import os
import pathlib

import requests
from flask import Flask, session, abort, redirect, request
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from pip._vendor import cachecontrol
import google.auth.transport.requests

# 'google_id': '110788464327696265201', Shuting Li, role: patient   # sl5185@columbia.edu
# 'google_id': '104405107080836112407', Shuting Li, role: doctor    # shuting.li.sli@gmail.com
# 'google_id': '117740487543455173970', Shuting Li, role: volunteer # lishuting.sli2@gmail.com


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

from keras.models import load_model

def extract_non_zero_indices(matrix):
    results = []
    for vector in matrix:
        # 获取非零元素的位置，但排除最后一个元素
        indices = [i+1 for i, val in enumerate(vector[:-1]) if val != 0.0]
        results.append(indices)
    return results



# 加载之前保存的模型
gru_model = load_model('best_gru_model.keras', custom_objects={'FixedAttentionLayer': FixedAttentionLayer})
lstm_model = load_model('best_lstm_model.keras', custom_objects={'FixedAttentionLayer': FixedAttentionLayer})
# Load the saved logistic regression model
import joblib
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
logreg_model = joblib.load('best_logreg_model.pkl')
rf_model = joblib.load('best_rf_model.pkl')

import xgboost as xgb
# Load the XGBoost model
bst = xgb.Booster()
bst.load_model('best_xgb_model.model')

app = Flask(__name__)

app.secret_key = "CodeSpecialist.com"
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
GOOGLE_CLIENT_ID = "41060034206-ommaaipvi81ap9cm3bq3neu702c2uger.apps.googleusercontent.com"
client_secrets_file = os.path.join(pathlib.Path(__file__).parent, "client_secret.json")

users_db = {
    '110788464327696265201': 'patient',
    '104405107080836112407': 'doctor',
    '117740487543455173970': 'volunteer',
}

flow = Flow.from_client_secrets_file(
    client_secrets_file=client_secrets_file,
    scopes=["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"],
    redirect_uri="http://127.0.0.1:5000/callback"
)


def login_is_required(function):
    def wrapper(*args, **kwargs):
        if "google_id" not in session:
            return abort(401)  # Authorization required
        else:
            return function()
    return wrapper

def call_microservices_sync():
    services = {
        "http://127.0.0.1:5000",  # Endpoint for Disease Prediction Service
        "http://127.0.0.1:5001",  # Endpoint for Appointment Scheduler Service
        # "http://127.0.0.1:5002": "/api/chat-response"  # Endpoint for AI Chat Service
    }
    responses = []
    for service, endpoint in services.items():
        full_url = service + endpoint
        response = requests.get(full_url)
        responses.append(f"Response from {service}{endpoint}: {response.text}")
    return responses

async def call_service(service):
    async with aiohttp.ClientSession() as session:
        async with session.get(service) as response:
            result = await response.text()
            return f"Response from {service}: {result}"

async def call_microservices_async():
    services = [
        "http://localhost:5000/login",
        "http://localhost:5001/login",
        # "http://localhost:5002"
    ]
    responses = []
    for _ in range(10):
        random.shuffle(services)
        responses += await asyncio.gather(*(call_service(service) for service in services))
    return responses

@app.route('/call-microservices-sync')
def sync_call_services():
    responses = call_microservices_sync()
    return jsonify(responses)

@app.route('/call-microservices-async')
def async_call_services():
    responses = asyncio.run(call_microservices_async())
    return jsonify(responses)

@app.route("/")
def navbar():
    print("main page accessed")
    print(session)
    if 'google_id' not in session:
        # 用户未登录，重定向到登录页面
        return redirect('/login')
    user_role = session.get('role', 'not logged in')
    return render_template('index.html', user_role=user_role)

@app.route("/login")
def login():
    print("login page accessed")
    authorization_url, state = flow.authorization_url(
        prompt='consent'  # Forces re-consent and re-authentication
    )
    print("authorization_url: ", authorization_url)
    session["state"] = state
    return redirect(authorization_url)


@app.route("/callback")
def callback():
    print("callback endpoint is accessed")
    flow.fetch_token(authorization_response=request.url)

    if not session["state"] == request.args["state"]:
        abort(500)  # State does not match!

    credentials = flow.credentials
    request_session = requests.session()
    cached_session = cachecontrol.CacheControl(request_session)
    token_request = google.auth.transport.requests.Request(session=cached_session)

    id_info = id_token.verify_oauth2_token(
        id_token=credentials._id_token,
        request=token_request,
        audience=GOOGLE_CLIENT_ID
    )

    session["google_id"] = id_info.get("sub")
    session["name"] = id_info.get("name")
    session['role'] = users_db.get(id_info.get("sub"), 'not logged in') # set user role
    return redirect("/protected_area")


@app.route("/logout")
def logout():
    print("logout endpoint accessed")

    # Optional: Revoke the Google token
    if 'credentials' in session:
        credentials = google.oauth2.credentials.Credentials(
            **session['credentials'])

        revoke = request.Request(
            'https://accounts.google.com/o/oauth2/revoke',
            params={'token': credentials.token},
            headers={'content-type': 'application/x-www-form-urlencoded'})

        try:
            request.urlopen(revoke)
        except Exception as e:
            print(f'An error occurred: {e}')

    session.clear()
    return redirect("/")

@app.route('/role')
def get_role():
    user_role = session.get('role', 'not logged in')
    return jsonify({"role": user_role})


@app.route('/api/is_logged_in')
def is_logged_in():
    is_logged_in = 'google_id' in session
    user_role = session.get('role', 'not logged in') if is_logged_in else 'not logged in'
    return jsonify(logged_in=is_logged_in, role=user_role)


@app.route("/protected_area")
def hello():
    if 'google_id' not in session:
        # 用户未登录，重定向到登录页面
        return redirect(url_for("login"))
    print("Accessed the protected area")
    print(session)
    return render_template('navbar.html')


@app.route("/model_output", methods=['GET', 'POST'])
def model_output():
    if request.method == 'POST':
        patient_id = int(request.form['patient_id'])
        index = patient_id - 1  # Convert patient_id to index

        # For GRU and LSTM
        for i, (x, y) in enumerate(data_generator(test_set_x, test_set_t, test_set_y, test_lengths)):
            if i == index:
                pred_y_gru = gru_model.predict_on_batch(np.reshape(x, (1,) + x.shape))[0][0]
                pred_y_lstm = lstm_model.predict_on_batch(np.reshape(x, (1,) + x.shape))[0][0]
                break

        # For Logistic Regression
        x_for_logreg, _ = prepare_data_for_logreg(test_set_x, test_set_y, test_lengths)
        x_single = x_for_logreg[index].reshape(1, -1)
        pred_y_logreg = logreg_model.predict_proba(x_single)[:, 1][0]

        # For Random Forest
        pred_y_rf = rf_model.predict_proba(x_single)[:, 1][0]
        # For XGBoost
        dtest_single = xgb.DMatrix(x_single)
        pred_y_xgb = bst.predict(dtest_single)[0]

        # Get the non-zero element positions
        input_data = extract_non_zero_indices(x.tolist())


        # Return the prediction results
        return render_template("model_output.html", predicted_output_gru=pred_y_gru, predicted_output_lstm=pred_y_lstm, predicted_output_logreg=pred_y_logreg, predicted_output_rf=pred_y_rf, predicted_output_xgb=pred_y_xgb, input_data=input_data, actual_output=y, patient_id=patient_id)
    else:
        return render_template("model_output.html")

@app.route("/labs")
def labs():
    return render_template("labs.html")

if __name__ == '__main__':
    app.run(debug=True)