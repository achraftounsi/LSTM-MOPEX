import multiprocessing
import os
import time
from pprint import pprint
import hydroeval as he
import pandas as pd
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from numpy import append
from numpy import array, reshape
from pandas import DataFrame, concat
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def add_steps(mrms: DataFrame, n_steps: int,_col='Q'):
    # split a univariate sequence into samples
    sequence = mrms
    sequence = list(sequence[_col])
    X = list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x = sequence[i:end_ix]
        X.append(seq_x)
    X = DataFrame(X)
    X.columns = [_col+str(e) for e in X.columns]
    return X



def load_data(_filename, n_steps = 4):
    data = pd.read_csv(os.path.join(ROOT_DIR, 'MOPEX_TS_Ach', _filename), date_parser=['datetime'],
                       parse_dates=True).set_index('datetime')
    data = data[data['Q'] >= 0]
    # choose a number of time steps
    X = add_steps(data, n_steps)
    # # split into samples
    mrms = data.iloc[:-n_steps, :].reset_index()
    mrms = concat([X, mrms], axis=1).set_index('datetime')
    mrms = mrms.drop(['Q0'], axis =1)
    return mrms


# def split_data(data):
#     train_x, test_x, train_y, test_y = train_test_split(data[[e for e in data.columns if e != 'Q']], data['Q'],
#                                                         train_size=0.7, test_size=0.3, shuffle=False)
#     return train_x, test_x, train_y, test_y

def split_data(data):
    train = data.iloc[:int(len(data)*0.8)]
    test = data.iloc[int(len(data)*0.8):]
    return train, test


def scale_data(data):
    # Scale the all of the data to be values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Splitting the data
    x_train = scaled_data[:, :len(data.columns) - 1]
    y_train = scaled_data[:, len(data.columns) - 1]
    # Convert to numpy arrays
    x_train, y_train = array(x_train), array(y_train)
    # Reshape the data into 3-D array
    x_train = reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler


def model_architect(data, units: int = 200, drp: float = 0.1):
    # Initialising the RNN
    model = Sequential()

    model.add(LSTM(units=units, return_sequences=True, input_shape=(data.shape[1]-1, 1)))
    model.add(Dropout(drp))

    # Adding a second LSTM layer and Dropout layer
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(drp))

    # # Adding a third LSTM layer and Dropout layer
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(drp))

    # # Adding a fourth LSTM layer and and Dropout layer
    model.add(LSTM(units=units))
    model.add(Dropout(drp))

    # Adding the output layer
    # For Full connection layer we use dense
    # As the output is 1D so we use unit=1
    model.add(Dense(units=1))
    return model


def compiler(model, x_train, y_train):  # , x_validate, y_validate):
    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 2  # arbitrary default
    # compile and fit the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    print('fitting ...')
    history = model.fit(
        x_train,
        y_train,
        epochs=100,
        batch_size=256,
        # validation_data=(x_validate, y_validate),
        workers=cpus,
        use_multiprocessing=True,
        verbose=2,
        shuffle=False,
    )
    return model, history


def modeler(train):
    """
    Compute the model architecture
    :param train: train data
    :param validate: validation data
    :return: compiled model and the validation loss history
    """

    # scale the train and test
    x_train, y_train, scaler = scale_data(train)
    # x_validate, y_validate, _scaler = scale_data(validate)

    # Initialize the model architecture
    model = model_architect(train, units=200, drp=0.1)

    # Train the model
    model, history = compiler(model, x_train, y_train)  # , x_validate, y_validate)

    return x_train, y_train, model, history


def predictor(test, model):
    """
    Compute the predictions
    :param test: test data
    :param model: trained model
    :return: the predicted test data
    """
    # Scale the data
    x_test, y_test, scaler = scale_data(test)
    # Predict the data
    y_pred = model.predict(x_test)

    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))
    y_pred = y_pred.reshape((y_pred.shape[0], y_pred.shape[1]))
    tmp = append(x_test, y_pred, 1)

    inversed = scaler.inverse_transform(tmp)
    real_y_pred_test = inversed[:, inversed.shape[1] - 1]
    test['pred_Q'] = real_y_pred_test

    return x_test, y_test, y_pred, test


def statistics(_time, y_test, y_pred):
    """
    Compute the statistics of the model
    :param _time: the lead time used for the data
    :param n_steps: number of steps used for the data
    :param y_test: true y data
    :param y_pred: predicted y data
    :return: dictionnary of statistics
    """
    kge, r, alpha, beta = he.evaluator(he.kge, y_pred, y_test)
    data = {
        'station': _filename.split('.')[0],
        'MAE': mean_absolute_error(y_test, y_pred),
        'R_squared': r2_score(y_test, y_pred),
        'MSE_squared': mean_squared_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred, squared=False),
        'NSE': he.evaluator(he.nse, y_pred, y_test)[0],
        'KGE': kge[0],
        'r': r[0],
        'Alpha': alpha[0],
        'Beta': beta[0]
    }

    return data

ress = pd.read_csv(os.path.join(ROOT_DIR,'output.csv'))
ress['station'] = ress['station'].apply(lambda x: x.split('/')[-1])
res = []
errors = []
for _filename in tqdm([os.path.join(ROOT_DIR, 'MOPEX_TS_Ach', e) for e in os.listdir(os.path.join(ROOT_DIR,'MOPEX_TS_Ach'))]):
    if _filename.split('/')[-1].split('.')[0] not in list(ress['station']):
        try:
            start = time.time()
            # _filename ='1048000.csv'
            data = load_data(_filename)
            train, test = split_data(data)
            x_train, y_train, model, history = modeler(train)
            x_test, y_test, y_pred, test = predictor(test, model)

            stats = statistics(_filename, test['Q'],test['pred_Q'])
            print(" Time taken for station ", time.time() - start)
            res.append(stats)

            DataFrame(res).to_csv(os.path.join(ROOT_DIR,'output.csv'))
        except:
            print("Error with station ", _filename.split('/')[-1])
            errors.append(_filename.split('/')[-1])
            pass