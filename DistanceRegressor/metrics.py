import numpy as np

# function to calculate absolute relative distance
def abs_relative_distance(d_pred, d_true):
    ard_array = np.absolute(d_pred - d_true)/d_true
    return ard_array

# function to calculate square relative distance
def sq_relative_distance(d_pred, d_true):
    sq_ard_array = np.absolute(d_pred - d_true)**2/d_true
    return np.mean(sq_ard_array)

# function to calculate root mean square error
def rmse(d_pred, d_true):
    rmse_array = np.absolute(d_pred - d_true)**2
    return np.sqrt(np.mean(rmse_array))

# function to calculate log rmse
def log_rmse(d_pred, d_true):
    log_rmse_array = np.absolute(np.log(d_pred)-np.log(d_true))**2
    return np.sqrt(np.mean(log_rmse_array))