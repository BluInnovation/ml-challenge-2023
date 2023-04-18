import numpy as np


def remove_outliers_iqr(arr, multiplier=1.5):
    # Calculate IQR for the given column
    q3 = np.nanpercentile(arr, 75)
    q1 = np.nanpercentile(arr, 25)
    iqr = q3 - q1
    # Calculate the lower and upper bounds
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    # Return the filtered array
    return arr[(arr > lower_bound) & (arr < upper_bound)]
