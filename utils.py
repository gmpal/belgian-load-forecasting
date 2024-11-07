import pandas as pd
import numpy as np
from datetime import datetime

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
def preprocess(filepath, lockdown=[(182692, 187181), (204772, 219268)]):
    data = pd.read_csv(filepath, delimiter=";", index_col='Datetime')[::-1]
    data.index = pd.to_datetime(data.index, errors='coerce', utc=True)
    data = data.drop(columns = ["Resolution code"])
    data["Year"] = [i.year for i in data.index]
    data["Month"] = [i.month for i in data.index]
    data["Day"] = [i.day for i in data.index]
    data["Hour"] = [i.hour for i in data.index]
    data["Minute"] = [i.minute for i in data.index]
    data["Season"] = data["Month"]//4
    data["Lockdown"] = 0
    for l in lockdown:
        data.iloc[l[0]:l[1], data.columns.get_loc("Lockdown")] = 1
    data['Total Load Interpolated'] = data['Total Load'].interpolate(method='linear')
    return data

def create_time_series_splits(data, 
                              train_size_days, 
                              test_size_days, 
                              num_splits, 
                              window_size_steps, 
                              exclude_columns, 
                              target_column, 
                              prediction_horizon_steps, 
                              shifting_steps=None,
                              elia_column_to_return=None):
    """
    Creates train/test splits for a multivariate time series dataset with maintained column names
    and datetime indices corresponding to the prediction time.
    
    Parameters:
    - data (pd.DataFrame): The input time series data with 15-minute granularity.
                           Assumes that data has a DateTimeIndex.
    - train_size_days (int): Number of days for the training set in each split.
    - test_size_days (int): Number of days for the testing set in each split.
    - num_splits (int): Number of non-overlapping train/test splits to generate.
    - window_size_steps (int): Number of 15-minute steps to consider for lagged values (window size).
    - exclude_columns (list): List of columns to exclude from the training features.
    - target_column (str): The name of the target column.
    - prediction_horizon_steps (int): Number of 15-minute steps ahead to predict (prediction horizon).
    - shifting_steps (int or None): Number of 15-minute steps to skip between samples.
                                    If None, defaults to 1 (no skipping).
    - elia_column_to_return (str or None): An additional column to return with the same format as Y (target column).
                                        Useful to compare the results of the model with the ELIA forecasting. 
                                        It can be for example 'Most recent forecast' or  'Day-ahead 6PM forecast'

    Returns:
    - splits (list): A list containing dictionaries with keys 'X_train', 'Y_train', 'X_test', 'Y_test' for each split.
                     Each DataFrame has its index set to the datetime right before prediction. 
                     If elia_column_to_return is not None, the dictionary will also contain 'ELIA_train' and 'ELIA_test'.
    """

    steps_per_day = 96  # Number of 15-minute intervals in a day
    train_size_steps = train_size_days * steps_per_day
    test_size_steps = test_size_days * steps_per_day

    if shifting_steps is None:
        shifting_steps = 1

    split_size_steps = train_size_steps + test_size_steps

    # Calculate the total number of data points required
    total_data_steps_needed = split_size_steps * num_splits + window_size_steps + prediction_horizon_steps - 1

    if len(data) < total_data_steps_needed:
        raise ValueError('Not enough data for the specified number of splits and sizes.')

    splits = []

    for split_index in range(num_splits):
        # Calculate indices for train and test data
        split_start = split_index * split_size_steps
        train_start = split_start
        train_end = train_start + train_size_steps - 1

        test_start = train_end + 1
        test_end = test_start + test_size_steps - 1

        # Extract train and test data for the current split
        train_data = data.iloc[train_start : train_end + window_size_steps + prediction_horizon_steps]
        test_data = data.iloc[test_start : test_end + window_size_steps + prediction_horizon_steps]

        # Generate training samples
        sample_train = create_samples_with_datetime_index(train_data, window_size_steps, exclude_columns, target_column, prediction_horizon_steps, shifting_steps, elia_column_to_return)

        # Generate testing samples
        sample_test = create_samples_with_datetime_index(test_data, window_size_steps, exclude_columns, target_column, prediction_horizon_steps, shifting_steps, elia_column_to_return)

        if elia_column_to_return:
            X_train, Y_train, ELIA_train = sample_train
            X_test, Y_test, ELIA_test = sample_test
            # Append the current split to the list
            splits.append({
                'X_train': X_train,
                'Y_train': Y_train,
                'X_test': X_test,
                'Y_test': Y_test,
                'ELIA_train': ELIA_train,
                'ELIA_test': ELIA_test
            })
        else:
            X_train, Y_train = sample_train
            X_test, Y_test = sample_test
            # Append the current split to the list
            splits.append({
                'X_train': X_train,
                'Y_train': Y_train,
                'X_test': X_test,
                'Y_test': Y_test
            })


    return splits

def create_samples_with_datetime_index(data, window_size_steps, exclude_columns, target_column, prediction_horizon_steps, shifting_steps, elia_column_to_return):
    """
    Generates samples for X and Y based on the window size and prediction horizon,
    maintaining column names with time step suffixes and setting the index to datetime.

    Parameters:
    - data (pd.DataFrame): The subset of data to create samples from.
                           Assumes that data has a DateTimeIndex.
    - window_size_steps (int): Number of steps in the window size.
    - exclude_columns (list): Columns to exclude from features.
    - target_column (str): The name of the target column.
    - prediction_horizon_steps (int): Number of steps ahead to predict.
    - shifting_steps (int): Number of steps to shift between samples.

    Returns:
    - X (pd.DataFrame): Feature DataFrame with maintained column names and datetime index.
    - Y (pd.DataFrame): Target DataFrame with datetime index.
    """

    # Ensure that the data index is a DateTimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have a DateTimeIndex.")

    # Determine feature columns by excluding specified columns and the target column
    feature_columns = [col for col in data.columns if col not in exclude_columns]

    max_t = len(data) - prediction_horizon_steps
    t_start = window_size_steps - 1
    t_end = max_t - 1

    t_values = list(range(t_start, t_end + 1, shifting_steps))

    X_list = []
    Y_list = []

    if elia_column_to_return: 
        ELIA_list = []
        
    index_list = []

    for t in t_values:
        # Initialize an empty dictionary to store the features for this sample
        X_t = {}
        
        # Iterate over the time window
        for w in range(window_size_steps):
            time_step = t - window_size_steps + w + 1
            suffix = f"_t-{window_size_steps - w - 1}"
            for col in feature_columns:
                col_name = f"{col}{suffix}"
                X_t[col_name] = data.iloc[time_step][col]
        
        # Extract prediction horizon for current sample
        Y_t = data.iloc[t + 1 : t + 1 + prediction_horizon_steps][target_column].values.flatten()
        if elia_column_to_return:
            ELIA_t = data.iloc[t + 1 : t + 1 + prediction_horizon_steps][elia_column_to_return].values.flatten()
            ELIA_list.append(ELIA_t)

        
        X_list.append(X_t)
        Y_list.append(Y_t)
        
        # Record the datetime index corresponding to time t (right before prediction at t+1)
        index_list.append(data.index[t])

    # Convert lists to DataFrames
    X = pd.DataFrame(X_list, index=index_list)
    Y = pd.DataFrame(Y_list, index=index_list, columns=[f"{target_column}_t+{i+1}" for i in range(prediction_horizon_steps)])

    if elia_column_to_return:
        ELIA = pd.DataFrame(ELIA_list, index=index_list, columns=[f"{elia_column_to_return}_t+{i+1}" for i in range(prediction_horizon_steps)])
        return X, Y, ELIA
    else:
        return X, Y