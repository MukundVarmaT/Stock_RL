import os
import math
import logging
import pandas as pd
import numpy as np
import keras.backend as K

format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))
format_currency = lambda price: '${0:.2f}'.format(abs(price))

def show_train_result(result, val_position, initial_offset):
    if val_position == initial_offset or val_position == 0.0:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), result[3]))
    else:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f})'
                     .format(result[0], result[1], format_position(result[2]), format_position(val_position), result[3],))
    
def get_stock_data(stock_file):
    df = pd.read_csv(stock_file)
    return list(df['Adj Close'])

def switch_k_backend_device():
    """ Switches `keras` backend from GPU to CPU if required.
    Faster computation on CPU (if using tensorflow-gpu).
    """
    if K.backend() == "tensorflow":
        logging.debug("switching to TensorFlow for CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"