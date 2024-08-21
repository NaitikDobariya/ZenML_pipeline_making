import logging
from typing import Tuple
from typing_extensions import Annotated

import pandas as pd
from zenml import step

from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStratrgy

@step
def clean_data(data_frame: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "Y_train"],
    Annotated[pd.Series, "Y_test"], 
]:
    
    try:
        process_strategy = DataPreProcessStratrgy()
        data_cleaning = DataCleaning(data_frame, process_strategy)
        processed_data = data_cleaning.handle_data()

        data_divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, data_divide_strategy)
        X_train, X_test, Y_train, Y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed")

        return X_train, X_test, Y_train, Y_test

    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e