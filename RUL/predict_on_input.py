import argparse
from IPython.display import display
import logging
import numpy as np
#from numpy import np.expm1, np.log1p, round, np.savetxt
import pandas as pd
from sklearn import metrics

from tensorflow.config import list_physical_devices
from keras.models import load_model

from utils import utils



def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse arguments',
                                     epilog='End parsing arguments')
    # --- input train and input_data
    parser.add_argument("--path_to_train_data", type=str, 
                        default="../data/train_FD001.txt",
                        help="path to the file with train data: \
                              used to check the drift in test data. \
                              If empty, no check will be done")
    #parser.add_argument("--dir_to_the_trained_artifacts", type=utils.dir_path, 
    #                    default="./models/",
    #                    help="directory to the saved baseline and model")
    parser.add_argument("--path_to_input_data", type=utils.file_path, 
                        default="../data/test_FD001.txt",
                        help="path to the file with test data")
    parser.add_argument("--path_to_RUL", type=str, 
                        default="../data/RUL_FD001.txt",
                        help="path to the file with test RUL. If empty, no evaluation will be done")

    # --- saved artifacts
    parser.add_argument("--trained_baseline", type=utils.file_path, 
                        default="./models/current_baseline.pkl",
                        help="trained baseline for data scaling")
    parser.add_argument("--trained_model", type=utils.dir_path, 
                        default="./models/current_model/",
                        help="trained model for RUL prediction")

    # --- output predictions
    parser.add_argument("--path_to_predictions", type=str, 
                        default="./predictions/current_prediction.csv",
                        help="path to the file for saving predictions")

    args = parser.parse_args()
    return args


def preprocess_input_data(input_data, baseline, window_size):
    logger.info('Scale sensors for test data')
    units = input_data['unit']
    bdata = utils.scale_sensors(input_data[utils.useful_sensors], baseline)

    logger.info('Shape data to window size')
    padding = utils.generate_padding_input(window_size)
    bdata_window = padding([bdata[units==unit].values 
                                     for unit in units.unique()])
    logger.debug(str(bdata_window.shape))
    return bdata_window


def predict_by_model(model, bdata_window):
    logger.info('by model')
    predi_logRUL = model.predict(bdata_window)
    predi_RUL = np.expm1(predi_logRUL)
    np.savetxt(args.path_to_predictions, predi_RUL, delimiter=",")
    return predi_logRUL

    
def read_RUL(path_to_RUL):
    logger.info(f'Reading Turbofan test RUL from {path_to_RUL:s}')
    RUL = pd.read_csv(path_to_RUL, sep=r"\s+", index_col=False, header=None)
    RUL.reset_index(inplace=True) # row number (starts from 0) serves as unit number
    RUL.columns=['unit', 'RUL']
    RUL['unit'] += 1 # the same unit numbers as in the input data
    if logger.isEnabledFor(logging.DEBUG):
        display(RUL)
    return RUL


def evaluate_predictions(predi_logRUL, RUL):    
    logger.info('Evaluating with test RUL')
    # assert there are as many RULs as test data units
    assert RUL.shape[0] == len(predi_logRUL), f'RUL.shape[0]{RUL.shape[0]:d}  len(predi_logRUL){len(predi_logRUL):d}'
    # log-scale    
    ground_truth_logRUL = RUL['RUL'].apply(np.log1p).values
    # calculate metrics
    mae_test = metrics.mean_absolute_error(y_pred = predi_logRUL.flatten(), 
                                           y_true=ground_truth_logRUL.flatten())
    r2_test = metrics.r2_score(y_pred = predi_logRUL.flatten(), 
                               y_true=ground_truth_logRUL.flatten())
    RUL['predicted'] = np.round(np.expm1(predi_logRUL),1)
    return mae_test, r2_test    
    
    
    
def this_main():
    logger.info(f'Reading input data from {args.path_to_input_data:s}')
    #input_data = read_input_data(args)
    colnames = ["unit", "cycle"] + ["setting"+str(i) for i in (1,2,3)] + \
               ["s"+str(i) for i in range(1,22)]
    input_data = pd.read_csv(args.path_to_input_data, sep=r"\s+", 
                             index_col=False, header=None, names=colnames)    


    # consistency check with train data
    if args.path_to_train_data:
        assert utils.file_path(args.path_to_train_data)
        logger.info('Reading train data and checking consistency')
        #train_data = read_train_data(args.path_to_train_data)
        train_data = pd.read_csv(args.path_to_train_data, sep=r"\s+", 
                             index_col=False, header=None, names=colnames) 
        
        train_q, input_q = utils.quantile_consistency(train_data, input_data, 
                                 sensors=None, 
                                 quantiles=[0.2, 0.4, 0.6, 0.8], rtol=0.1)
        if logger.isEnabledFor(logging.DEBUG):
            display(train_q); display(input_q)

    logger.info('Load baseline')
    baseline = pd.read_pickle(args.trained_baseline)

    logger.info('Load keras model')
    model = load_model(args.trained_model)        
    if logger.isEnabledFor(logging.INFO):
        display(model.summary())
    logger.info('Get the shape of the model input to read the window_size')
    _, window_size, n_features = model.input_shape
    assert len(utils.useful_sensors) == n_features
        
    logger.info('Predict on input data')
    bdata_window =  preprocess_input_data(input_data, baseline, window_size)
    predi_logRUL = predict_by_model(model, bdata_window)
        
    # Perform evaluation if RUL data are available
    if args.path_to_RUL:
        RUL = read_RUL(args.path_to_RUL)
        
        mae_test, r2_test = evaluate_predictions(predi_logRUL, RUL)
        print('MAE test {:.3f}'.format(mae_test))
        print('R2 test {:.3f}'.format(r2_test))
    
    
if __name__=='__main__':
    
    args = parse_arguments()
    
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s %(levelname)-8s %(name)-12s %(funcName)s line_%(lineno)d %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    logging.info('Check GPU for tensorflow')
    logging.info(list_physical_devices('GPU'))

    this_main()
    
    logger.info('DONE processing predict_on_test.py')
        