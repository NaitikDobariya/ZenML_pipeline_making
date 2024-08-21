import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):

    @abstractmethod
    def calculate_scores(self, Y_true: np.ndarray, Y_pred: np.ndarray):
        pass


class MSE(Evaluation):

    def calculate_scores(self, Y_true: np.ndarray, Y_pred: np.ndarray):
        
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(Y_true, Y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e
        
class R2(Evaluation):

    def calculate_scores(self, Y_true: np.ndarray, Y_pred: np.ndarray):
        
        try:
            logging.info("Calculating MSE")
            r2 = r2_score(Y_true, Y_pred)
            logging.info("MSE: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e
    
class RMSE(Evaluation):

    def calculate_scores(self, Y_true: np.ndarray, Y_pred: np.ndarray):
        
        try:
            logging.info("Calculating MSE")
            rmse = mean_squared_error(Y_true, Y_pred, squared = False)
            logging.info("MSE: {}".format(rmse))
            return rmse 
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e