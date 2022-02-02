import argparse
from datetime import datetime
from IPython.display import display
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
from numpy import log1p
import os
import pandas as pd
import shutil

from tensorflow.config import list_physical_devices
from tensorflow.keras import callbacks, optimizers

from utils import utils



        
def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse arguments',
                                     epilog='End parsing arguments')
    # --- input train and test_data
    parser.add_argument("--path_to_train_data", type=utils.file_path, default="../data/train_FD001.txt",
                        help="path to the file with train data")
    parser.add_argument("--dir_to_the_tensorboard_logs", type=utils.dir_path, default="../logs/",
                        help="directory to save the trained baseline and the trained model")
    parser.add_argument("--dir_to_the_trained_artifacts", type=utils.dir_path, default="./models/",
                        help="directory to save the trained baseline and the trained model")
    # --- business-choice-parameters
    parser.add_argument("--window_size", type=int, default=32,
                        help="how many data points are available for each unit to make predictions")

    # --- model-related parameters
    parser.add_argument("--dimred", type=int, default=2,
                        help="number of units in the 1st GRU layer to ensure dimensionality reduction, reasonable values are 2,3,4")
    parser.add_argument("--gru_units", type=int, default=6,
                        help="number of units in the 2nd GRU layer")

    # --- model-training-related parameters
    parser.add_argument("--epochs", type=int, default=100,
                        help="epochs for training the model")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size for training the model")
    # --- hash for unique names of the saved artifacts
    parser.add_argument("--hash_timestamp", type=str, default=f'{datetime.today():%Y-%m%d-%H%M}',
                        help="unique string to add to all saved artifacts, \
                        defaut is the current year-month-data-hour-minute")
    
    args = parser.parse_args()
    return args
    
    
def prepare_train_data(data):
    logger.info('calculating RUL')
    max_cycle = data["cycle"].groupby(data['unit']).transform('max')
    data["RUL"] = max_cycle - data["cycle"]
    # assert that data are sorted as expected
    assert all(data.sort_values(by=['unit', 'RUL'], ascending=[True, False])== data)

    logger.info('removing useless columns')
    data = data[[c for c in data.columns if c not in (utils.useless_sensors+utils.factor_sensors+utils.settings)]]
    if logger.isEnabledFor(logging.DEBUG):
        display(data.head())

    logger.info('calculating baseline')
    baseline = utils.get_baseline(data)
    if logger.isEnabledFor(logging.DEBUG):
        display(baseline)

    baseline_filename = os.path.join(args.dir_to_the_trained_artifacts, f'{args.hash_timestamp:s}_baseline.pkl')
    logger.info(f'saving baseline to {baseline_filename:s}')
    baseline.to_pickle(baseline_filename)
    shutil.copy(baseline_filename, os.path.join(args.dir_to_the_trained_artifacts, 'current_baseline.pkl'))                       
        
    return data, baseline



def build_and_train_model(args, data, baseline):
    logger.info('Build and train model')
    logger.debug('Scale sensors')
    bdata = utils.scale_sensors(data, baseline)    
    logger.debug('keep unit and RUL in separate series')
    bdata_unit = data['unit'].copy()
    bdata_RUL = data['RUL'].copy()
    logger.debug('log-transform RUL')
    bdata_logRUL = bdata_RUL.apply(log1p) 

    logger.info('build model')
    gru1 = utils.make_gru_seq_model(seq_length=args.window_size,  
                             n_features=bdata.shape[1], # number of useful sensors
                             dimred=args.dimred, 
                             gru_units=args.gru_units)
    if logger.isEnabledFor(logging.DEBUG):
        display(gru1.summary())

    logger.info('compiling model')
    gru1.compile(optimizer=optimizers.Adam(1e-3), 
                 loss='mean_squared_error', metrics=['mae'])
    
    logger.info('prepare Tensorboard and callbacks')
    monitor = 'mae'
    
    model_spec = f'{args.hash_timestamp:s}_window{args.window_size:d}_dimred{args.dimred:d}_gru{args.gru_units:d}'
    log_dir = os.path.join(args.dir_to_the_tensorboard_logs, model_spec)
    tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=50, write_graph=False,
                                           write_grads=False, write_images=True)

    logger.info('I train on the whole dataset without validation, monitoring on train loss')
    reduce_lr = callbacks.ReduceLROnPlateau(monitor=monitor,
                                            factor=0.5, patience=10, min_delta=0.001)

    early_cb = callbacks.EarlyStopping(monitor=monitor, 
                                       min_delta=0.001, patience=15, 
                                       restore_best_weights=True)
    
    model_filename = os.path.join(args.dir_to_the_trained_artifacts, model_spec)
    logger.info(f'Best model will be saved to {model_filename:s}')
    best_cb = callbacks.ModelCheckpoint(filepath=model_filename, monitor=monitor,
                                        save_freq='epoch', save_best_only=True)

    
    
    logger.info('Generator for input data using all units from training data')
    data_gen = utils.gen_random_sequences_various_units(unit_se=bdata_unit, df=bdata, target_se=bdata_logRUL, 
                                                  seq_length=args.window_size, 
                                                  selected_units=bdata_unit.tolist(), batch_size=args.batch_size)

    logger.info('Fit model on training data')
    hist = gru1.fit(data_gen, epochs=args.epochs, steps_per_epoch=60, 
                    callbacks=[tensorboard_cb, early_cb, reduce_lr, best_cb], 
                    verbose=1)

    logger.info('Show performance on training data')
    performance_se = pd.Series(index=hist.history.keys(), dtype=float)
    for key in hist.history.keys():
        performance_se.loc[key] = hist.history[key][-1]
    if logger.isEnabledFor(logging.INFO):
        display(performance_se)

    logger.info(f'Copy model to the current model in {args.dir_to_the_trained_artifacts:s}')
    shutil.copytree(model_filename, os.path.join(args.dir_to_the_trained_artifacts, 'current_model'), dirs_exist_ok=True)                   
    
    return hist, gru1

    
                           


def this_main():    
    # Read training data
    logger.info(f'Reading Turbofan train data from {args.path_to_train_data:s}')
    colnames = ["unit", "cycle"] + ["setting"+str(i) for i in (1,2,3)] + \
               ["s"+str(i) for i in range(1,22)]
    train_data = pd.read_csv(args.path_to_train_data, sep=r"\s+", 
                             index_col=False, header=None, names=colnames)    

    # Prepare data for training, calculate and save baseline
    df, baseline = prepare_train_data(train_data)

    # Build, train and save model                            
    hist, gru1 = build_and_train_model(args, df, baseline)                           

    return None
    
    
    
if __name__=='__main__':

    args = parse_arguments()

    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s %(levelname)-8s %(name)-12s %(funcName)s line_%(lineno)d %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    logging.info('Check GPU for tensorflow')
    logging.info(list_physical_devices('GPU'))

    this_main()
    logger.info('DONE processing train_model.py')
    
