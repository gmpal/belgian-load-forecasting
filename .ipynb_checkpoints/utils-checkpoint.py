import pandas as pd
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm

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

from multiprocessing import Pool

def process_split(args):
    (split_index, data, train_size_steps, test_size_steps, window_size_steps, exclude_columns,
     target_column, prediction_horizon_steps, shifting_steps, elia_column_to_return, alignment_times) = args

    split_size_steps = train_size_steps + test_size_steps

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
    sample_train = create_samples_with_datetime_index(
        train_data, window_size_steps, exclude_columns, target_column,
        prediction_horizon_steps, shifting_steps, elia_column_to_return, alignment_times=None
    )

    # Generate testing samples
    sample_test = create_samples_with_datetime_index(
        test_data, window_size_steps, exclude_columns, target_column,
        prediction_horizon_steps, shifting_steps, elia_column_to_return, alignment_times
    )

    if elia_column_to_return:
        X_train, Y_train, ELIA_train = sample_train
        X_test, Y_test, ELIA_test = sample_test
        return {
            'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test,
            'Y_test': Y_test, 'ELIA_train': ELIA_train, 'ELIA_test': ELIA_test
        }
    else:
        X_train, Y_train = sample_train
        X_test, Y_test = sample_test
        return {
            'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test
        }

def create_time_series_splits(data, train_size_days, test_size_days, num_splits,
                              window_size_steps, exclude_columns, target_column,
                              prediction_horizon_steps, shifting_steps=None,
                              elia_column_to_return=None, alignment_times=None, n_jobs=1):
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
    - alignment_times (list or None): A list of times to align the samples to. Should run with shifting_steps = 1, it will select a subset of timestep that aligns with the specified times. (To start testing set at 6pm for example)

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
    total_data_steps_needed = (split_size_steps * num_splits +
                               window_size_steps + prediction_horizon_steps - 1)

    if len(data) < total_data_steps_needed:
        raise ValueError('Not enough data for the specified number of splits and sizes.')

    # Prepare arguments for each split
    args_list = [
        (split_index, data, train_size_steps, test_size_steps, window_size_steps,
         exclude_columns, target_column, prediction_horizon_steps, shifting_steps,
         elia_column_to_return, alignment_times)
        for split_index in range(num_splits)
    ]

    # Use multiprocessing Pool to process splits in parallel
    with Pool(n_jobs) as pool:
        splits = pool.map(process_split, args_list)

    return splits


# def create_time_series_splits(data, 
#                               train_size_days, 
#                               test_size_days, 
#                               num_splits, 
#                               window_size_steps, 
#                               exclude_columns, 
#                               target_column, 
#                               prediction_horizon_steps, 
#                               shifting_steps=None,
#                               elia_column_to_return=None,
#                               alignment_times=None):
#     """
#     Creates train/test splits for a multivariate time series dataset with maintained column names
#     and datetime indices corresponding to the prediction time.
    
#     Parameters:
#     - data (pd.DataFrame): The input time series data with 15-minute granularity.
#                            Assumes that data has a DateTimeIndex.
#     - train_size_days (int): Number of days for the training set in each split.
#     - test_size_days (int): Number of days for the testing set in each split.
#     - num_splits (int): Number of non-overlapping train/test splits to generate.
#     - window_size_steps (int): Number of 15-minute steps to consider for lagged values (window size).
#     - exclude_columns (list): List of columns to exclude from the training features.
#     - target_column (str): The name of the target column.
#     - prediction_horizon_steps (int): Number of 15-minute steps ahead to predict (prediction horizon).
#     - shifting_steps (int or None): Number of 15-minute steps to skip between samples.
#                                     If None, defaults to 1 (no skipping).
#     - elia_column_to_return (str or None): An additional column to return with the same format as Y (target column).
#                                         Useful to compare the results of the model with the ELIA forecasting. 
#                                         It can be for example 'Most recent forecast' or  'Day-ahead 6PM forecast'
#     - alignment_times (list or None): A list of times to align the samples to. Should run with shifting_steps = 1, it will select a subset of timestep that aligns with the specified times. (To start testing set at 6pm for example)

#     Returns:
#     - splits (list): A list containing dictionaries with keys 'X_train', 'Y_train', 'X_test', 'Y_test' for each split.
#                      Each DataFrame has its index set to the datetime right before prediction. 
#                      If elia_column_to_return is not None, the dictionary will also contain 'ELIA_train' and 'ELIA_test'.
#     """

#     steps_per_day = 96  # Number of 15-minute intervals in a day
#     train_size_steps = train_size_days * steps_per_day
#     test_size_steps = test_size_days * steps_per_day

#     if shifting_steps is None:
#         shifting_steps = 1

#     split_size_steps = train_size_steps + test_size_steps

#     # Calculate the total number of data points required
#     total_data_steps_needed = split_size_steps * num_splits + window_size_steps + prediction_horizon_steps - 1

#     if len(data) < total_data_steps_needed:
#         raise ValueError('Not enough data for the specified number of splits and sizes.')

#     splits = []

#     for split_index in range(num_splits):
#         # Calculate indices for train and test data
#         split_start = split_index * split_size_steps
#         train_start = split_start
#         train_end = train_start + train_size_steps - 1

#         test_start = train_end + 1
#         test_end = test_start + test_size_steps - 1

#         # Extract train and test data for the current split
#         train_data = data.iloc[train_start : train_end + window_size_steps + prediction_horizon_steps]
#         test_data = data.iloc[test_start : test_end + window_size_steps + prediction_horizon_steps]

#         # Generate training samples
#         sample_train = create_samples_with_datetime_index(train_data, window_size_steps, exclude_columns, target_column, prediction_horizon_steps, shifting_steps, elia_column_to_return,alignment_times=None) # training set does not require alignment

#         # Generate testing samples
#         sample_test = create_samples_with_datetime_index(test_data, window_size_steps, exclude_columns, target_column, prediction_horizon_steps, shifting_steps, elia_column_to_return,alignment_times)

#         if elia_column_to_return:
#             X_train, Y_train, ELIA_train = sample_train
#             X_test, Y_test, ELIA_test = sample_test
#             # Append the current split to the list
#             splits.append({
#                 'X_train': X_train,
#                 'Y_train': Y_train,
#                 'X_test': X_test,
#                 'Y_test': Y_test,
#                 'ELIA_train': ELIA_train,
#                 'ELIA_test': ELIA_test
#             })
#         else:
#             X_train, Y_train = sample_train
#             X_test, Y_test = sample_test
#             # Append the current split to the list
#             splits.append({
#                 'X_train': X_train,
#                 'Y_train': Y_train,
#                 'X_test': X_test,
#                 'Y_test': Y_test
#             })


#     return splits

def create_samples_with_datetime_index(data, window_size_steps, exclude_columns, target_column, prediction_horizon_steps, shifting_steps, elia_column_to_return,alignment_times):
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

    # Convert alignment_times to a set of datetime.time objects
    if alignment_times is not None:
        alignment_times = {pd.to_datetime(t).time() if isinstance(t, str) else t for t in alignment_times}


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

    for t in tqdm(t_values):
        # Initialize an empty dictionary to store the features for this sample
        timestamp_t = data.index[t]
        # If alignment_times is specified, check if the time of timestamp_t matches
        if alignment_times is not None and timestamp_t.time() not in alignment_times:
            continue  # Skip this time step if it doesn't match the alignment times


        X_t = {}
        
        # Iterate over the time window
        for col in feature_columns:
            for w in range(window_size_steps):
                time_step = t - window_size_steps + w + 1
                suffix = f"_t-{window_size_steps - w - 1}"
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
    

def construct_iterative_forecast(X_test, trained_model, starting_time = '18:00', number_of_steps = 30, window_size_steps=4, target_column = 'Total Load Interpolated', granularity_minutes = 15, verbose=False): 
    """
    Generates forecasts iteratively for a specified number of steps ahead, updating the test data with past predictions at each step.

    This function simulates real-time forecasting by updating the test dataset (`X_test`) with predictions from the previous steps,
    mimicking a scenario where future inputs are partially dependent on past outputs. It is particularly useful for time series forecasting
    tasks where predictions are sequentially dependent.

    Parameters:
    -----------
    X_test : pandas DataFrame
        The test dataset containing the features for forecasting. Must have a datetime-like index.
    trained_model : model object
        The pre-trained model capable of making predictions. This model must have a `predict` method.
    starting_time : str, optional
        The starting time as a string in 'HH:MM' format, from which the forecasting will begin. Default is '18:00'.
    number_of_steps : int, optional
        The total number of forecasting steps to perform. Default is 30.
    window_size_steps : int, optional
        The number of past steps to use for updating the test dataset at each forecasting step. Default is 4.
    target_column : str, optional
        The name of the target variable column in `X_test` that needs to be forecasted and updated. Default is 'Total Load Interpolated'.
    granularity_minutes : int, optional
        The time interval in minutes between each forecasting step. Default is 15 minutes.
    verbose : bool, optional
        If set to True, prints detailed debug information at each step of the forecasting process. Default is False.

    Returns:
    --------
    predictions : pandas DataFrame
        A DataFrame containing the predictions for each forecasting step, indexed similarly to the input `X_test`.

    Examples:
    ---------
    >>> preds = construct_iterative_forecast(X_test, trained_model, starting_time='18:00', number_of_steps=96, window_size_steps=4, target_column='Total Load Interpolated', granularity_minutes=15, verbose=True)
    >>> print(preds.head())
    """

    predictions = pd.DataFrame(dtype='float64')
    for forecasting_step in range(number_of_steps):
        if verbose: 
            print(f'Forecasting step {forecasting_step}')
        
        current_t = pd.to_datetime(starting_time) + pd.Timedelta(minutes=granularity_minutes * (forecasting_step)) # one step before forecasting time
        if verbose:
            print(f'Current time: {current_t}')
        
        X_test_current = X_test.loc[X_test.index.time == current_t.time()]

        if forecasting_step <= window_size_steps:
            to_update = forecasting_step
        else:
            to_update = window_size_steps

        if verbose:
            print(f'Updating {to_update} columns')

        # only some columns need to be updated 
        for i in range(to_update):
            previous_time = pd.to_datetime(current_t) - pd.Timedelta(minutes=granularity_minutes * (i + 1))
            if verbose:
                print(f'Updating column {target_column}_t-{i} with past predictions at time {previous_time}')
            past_predictions = predictions.loc[predictions.index.time == previous_time.time()]
            if verbose:
                print('Past predictions')
                print(past_predictions.head())
            X_test_current.loc[:,f'{target_column}_t-{i}'] = past_predictions.values.flatten().astype('float64')

        if verbose:
            print('Updated X_test_current')
            print(X_test_current.head())

        Y_pred_current = pd.DataFrame(trained_model.predict(X_test_current), index=X_test_current.index, columns=[target_column], dtype='float64')
        
        if verbose: 
            print("Predictions")
            print(Y_pred_current.head())

        predictions = pd.concat([predictions, Y_pred_current])

        if verbose:
            print('\n\n')


    return predictions