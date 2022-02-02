import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(name)-12s %(funcName)s line_%(lineno)d %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

useless_sensors = ["s" + str(i)  for i in [1,5,10,16,18,19]]
factor_sensors = ["s6"]
numeric_sensors = ["s" + str(i)  for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
useful_sensors = numeric_sensors
settings = ["setting1", "setting2", "setting3"]


# --------------------------------------------  for argparse
def file_path(string):
    """ check if the path to the file"""
    logger.debug(string)
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)

def dir_path(string):
    """ check if the path to the file"""
    logger.debug(string)
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
        

        
# ------------------------------  consistency check train and test data

def quantile_consistency(train, test, sensors=None, 
                         quantiles=(0.2, 0.4, 0.6, 0.8), rtol=0.1):
    if sensors is None:
        sensors = useful_sensors
        
    train_q = train[sensors].quantile(q=quantiles, axis=0)
    train_q.index.name = 'train_quantiles'
        
    test_q = test[sensors].quantile(q=quantiles, axis=0)
    test_q.index.name = 'test_quantiles'
        
    pd.testing.assert_frame_equal(train_q, test_q, check_names=False, 
                                  check_exact=False, rtol=rtol)
    
    return train_q, test_q
        

# -------------------------------------------   for data preparation

# def remove_baseline_each_unit(data, start_steps=30):
#     """
#     Removes baseline of each unit separately. 
#     Only applicable if data for the normal operation of each unit are available
#     Data are supposed to be sorted chtonologically
#     bdata = remove_baseline_each_unit(data)
#     """
    
#     def _remove(df):
#         baseline = df[0:start_steps].mean()
#         return (df - baseline)/baseline * 100

#     bdata = data.groupby(data["unit"]).transform(_remove)
    
#     return bdata

def get_baseline(data, max_cycle=30, top_RUL=0.05):
    if 'cycle' in data.columns:
        # assume that each unit in training data has 30 
        return data.loc[data.cycle<max_cycle, useful_sensors].mean()
    elif 'RUL' in data.columns:
        assert 0 < top_RUL <= 0.2, 'top_RUL should be at most 0.2'
        # use top-5% RULs as baselines
        threshold_RUL = data['RUL'].quantile(1-top_RUL)
        return data.loc[data.RUL>threshold_RUL, useful_sensors].mean()
    else:
        raise KeyError('cannot calculate baseline: neither "Cycle" nor "RUL" column is data')
        return None

def scale_sensors(data, baseline):
    return (data[useful_sensors].div(baseline, axis=1)-1)*100 # to_percent



# ---------------------------------------------------  generate samples for Deep Learning

def gen_random_sequences_various_units(unit_se, df, target_se, seq_length=16, batch_size=32, 
                                selected_units=None, verbose=False):
    """
    Generates batches to use for model training or validation
    Inputs:
        unit_se: pd.Series -  unit numbers extracted from train data
        df: pd.DataFrame - data frame with scaled data
        target_se: pd.Series - logRUL extracted from the train data
        seq_length: int - sequence length for the model input
        batch_size: int - size of the batch size to generate
        selected_units: list - list of units to use, it can be train_units, validation_units 
    
    Output:
        xx: (batch_size, seq_length, number_of_columns_in_df) - batch of features
        yy: (batch_size,) - ground truth for RUL
    """

    
    assert len(df.shape) == 2
    probs = np.exp(-0.5*target_se[unit_se.isin(selected_units)].values); probs = probs/sum(probs)
    tmp = np.hstack([unit_se[unit_se.isin(selected_units)].values.reshape(-1,1),
                     target_se[unit_se.isin(selected_units)].values.reshape(-1,1),
                     df[unit_se.isin(selected_units)].values])
    #display(tmp[:5,:])
    
    while True:
        xx = list()
        yy = list()
        while len(yy)<batch_size:
            stop = np.random.choice(range(len(tmp)), 1, p=probs)[0]
            start = stop - seq_length
            if tmp[start,0] == tmp[stop,0]: # all data belong to one unit
                xx.append(tmp[start:stop,2:])
                yy.append(tmp[stop,1])
                if verbose:
                    print('unit {:.0f}, start {:d}, stop {:d}, RUL {:.0f}'.format(
                           tmp[stop,0], start, stop, np.expm1(tmp[stop,1])) )
            
        yield np.stack(xx), np.stack(yy)


## ----- I do not remember anymore what it was for in the SimplestGRU.ipynb
## -----
# def gen_random_sequences_from2D(unit_se, df, target, seq_length=16, batch_size=32, 
#                                 selected_units=None, verbose=False):
#     assert len(df.shape) == 2
#     if selected_units is None:
#         units = bdata_unit.unique().tolist()
#     else:
#         units = list(selected_units)
    
#     while True:
#         # each batch selects sequencies from one unit
#         unit = random.sample(units, 1)[0]
#         #print(f'unit {unit:d}')
#         n = sum(unit_se==unit)
#         # ----- select data close to unit dying with higher probability
#         possible_stops = np.array(range(seq_length, n))
#         probs = np.exp(-0.03*(n-possible_stops))
#         probs = probs/sum(probs)
        
#         unit_data = df.loc[unit_se==unit,:]
#         target_data = target.loc[unit_se==unit]
#         xx = list()
#         yy = list()
#         for stop in np.random.choice(possible_stops, batch_size, p=probs):
#             start = stop - seq_length
#             if verbose:
#                 print('unit {:d}, start {:d}, stop {:d}'.format(unit, start, stop) )
#             xx.append(unit_data.iloc[start:stop,:])
#             yy.append(target_data.iloc[stop])
#         yield np.stack(xx), np.stack(yy)
#
## ---- and this is how it was tested - and it WORKS
# data_gen = gen_random_sequences_from2D(unit_se=bdata_unit, df=bdata, target=bdata_logRUL, 
#                                        seq_length=WINDOW_SIZE, selected_units=units_tvt['train'],
#                                        batch_size=8, verbose=True)
## ---- 
# xx, yy = next(data_gen)
# print('shape of xx', xx.shape)
# print('shape of yy', yy.shape, 'yy', yy)



# ---------------------------------------------------  model training  for Deep Learning

def make_gru_seq_model(seq_length, n_features, dimred=2, gru_units=8):
    input_sensors = layers.Input(shape=(seq_length, n_features), 
                                 name='input_sensors')
    x = layers.GRU(units=dimred, return_sequences=True, 
                   stateful=False, name='dimred')(input_sensors)
    #x = layers.Dense(dimred, activation=None, name='dense1')(input_sensors)
    x = layers.GRU(units=gru_units, return_sequences=False, stateful=False, name='gru1')(x)
    #x = layers.gru(units=64, return_sequences=True, name='gru2')(x)
    RUL = layers.Dense(1, activation=None, name='RUL')(x)
    gru1 = models.Model(inputs=input_sensors, outputs=RUL)
    return gru1



def asymmetric_loss_function(alpha=0.5):
    def lo(x):
        return np.exp(alpha*x) - alpha*x -1
    return lo

def asymmetric_loss(y_true, y_pred):
    """ Asymmetric loss with higher penalty for overestimation of y
    """
    alpha=1.0
    x = y_pred-y_true 
    # positive x are desaster, units live shorter as expected
    # small negative x are OK, units live longer as expected
    f = K.exp(alpha*x) - alpha*x -1
    asy_loss = K.mean(f)
    return asy_loss

def R2(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# ----------------------------------------------------- for prediction on test set

from keras.preprocessing.sequence import pad_sequences
def generate_padding_input(WINDOW_SIZE):
    def padding_input(sequences):
        #sequence1 = np.expand_dims(sequence,0)
        return pad_sequences(sequences, maxlen=WINDOW_SIZE, dtype=float, 
                            padding='pre', truncating='pre', value=0.0)
    return padding_input    




# ------------------------------------------------------  for visulaizations
def scatter_log_scale(absciss, ordinate, figsize=(8,8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(absciss, ordinate, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('predicted log1pRUL')
    ax.set_ylabel('ground truth log1pRUL')
    #ax.set_xlim([-1,10])
    #ax.set_ylim([-1,10])
    ax.plot([0,5], [0, 5], color='green', lw=2)
    ax.grid(True)
    #display(fig)
    #plt.close(fig)
    return fig, ax

def scatter_linear_scale(absciss, ordinate, figsize=(8,8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(np.expm1(absciss), np.expm1(ordinate), alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('predicted RUL')
    ax.set_ylabel('ground truth RUL')
    ax.set_xlim([-1,50])
    ax.set_ylim([-1,50])
    ax.plot([0,120], [0, 120], color='green', lw=2)
    ax.grid(True)

    #display(fig)
    #plt.close(fig)
    return fig, ax

def scatter_linear_scale_lowRUL(absciss, ordinate, figsize=(8,8),
                                suptitle='Grund-Truth versus Predicted RUL close to unit death'):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(np.expm1(absciss), np.expm1(ordinate), alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('predicted RUL')
    ax.set_ylabel('ground truth RUL')
    ax.set_xlim([-1,9])
    ax.set_ylim([-1,9])
    ax.grid()
    ax.plot([0,10], [0, 10], color='green', lw=2)
    #display(fig)
    #plt.close(fig)
    ax.grid()
    ax.set_title(suptitle)
    return fig, ax


def scatter_groundtruth_vs_predicted(absciss, ordinate, figsize=(15,7), xy_lim=(-1,50), xy_low_lim = (-1,10),
                                suptitle='Grund-Truth versus Predicted RUL'):
    
    fig, ax = plt.subplots(1,2, figsize=figsize)
    
    ax[0].scatter(np.expm1(absciss), np.expm1(ordinate), alpha=0.3)
    ax[0].set_aspect('equal')
    ax[0].set_xlabel('predicted RUL')
    ax[0].set_ylabel('ground truth RUL')
    ax[0].set_xlim(xy_lim)
    ax[0].set_ylim(xy_lim)
    ax[0].plot([0,200], [0, 200], color='green', lw=2)
    ax[0].grid(True)
    
    ax[1].scatter(np.expm1(absciss), np.expm1(ordinate), alpha=0.3)
    ax[1].set_aspect('equal')
    ax[1].set_xlabel('predicted RUL')
    ax[1].set_ylabel('ground truth RUL')
    ax[1].set_xlim(xy_low_lim)
    ax[1].set_ylim(xy_low_lim)
    ax[1].plot([0,100], [0, 100], color='green', lw=2)
    ax[1].grid(True)
    ax[1].set_title('close to unit death')

    return fig


