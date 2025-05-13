import numpy as np
import pandas as pd


def range_filter(df, lower_bound, upper_bound, invalid_flag=-9999):
    """
    Filter a dataframe by a given lower and upper bound.

    Args:
        df (pandas.DataFrame): The dataframe to filter.
        lower_bound (float): The lower bound for the filter.
        upper_bound (float): The upper bound for the filter.
        invalid_flag (float): The flag value to replace invalid values with.

    Returns:
        pandas.DataFrame: The filtered dataframe.
    """

    # Apply the filter using numpy.where
    filtered_df = np.where((df >= lower_bound) & (df <= upper_bound), df, invalid_flag)

    # Convert the filtered data back to a DataFrame
    filtered_df = pd.DataFrame(filtered_df, index=df.index, columns=df.columns)

    # Initialize the range filtered stats dictionary
    range_filtered_stats = {}

    # Calculate the range filter stats
    for col in filtered_df.columns:
        num_flagged = (filtered_df[col] == invalid_flag).sum()
        percent_flagged = num_flagged / len(filtered_df[col]) * 100
        range_filtered_stats[col] = {
            "percent_flagged": percent_flagged,
            "num_flagged": num_flagged,
        }

    return filtered_df, range_filtered_stats

    return filtered_df


def create_diff_data(data, forward=True, diff_num=1):
    """
    Returns a dataframe that has been differenced the specified number of times in the specified direction.

    Args:
        data (pandas.DataFrame): The input data to difference.
        forward (bool, optional): Determines whether to difference the data forwards (True) or backwards (False).
            Defaults to True.
        diff_num (int, optional): The number of times to difference the data. Defaults to 1.

    Returns:
        pandas.DataFrame: The differenced data.
    """

    diff_data = data.copy()
    diff_data = data.sort_index(ascending=forward)

    for i in range(diff_num):
        diff_data = diff_data.diff()

    return diff_data


def gradient_filter(
    data,
    change_threshold,
    upper_bound,
    lower_bound,
    repeat_threshold,
    gradient_flag_value=-9990,
    diff_depth=2,
    margin=3,
    time_interval="10T",
):
    """
    Applies gradient filtering to the given data, returning a fully flagged dataset.
    Gradient filtering identifies sections of the data where values compared to previous and
    subsequent data points over a rolling window of time.If the change in gradient is below a
    certain threshold, and lasts for a threshold number of data points the data is flagged.
    Args:
        data (pandas.DataFrame): The data to filter.
        change_threshold (float): The threshold for change to trigger flagging.
        upper_bound (float): The upper bound for data values.
        lower_bound (float): The lower bound for data values.
        repeat_threshold (int): The number of times a change in gradient must be observed in the rolling window
            for the data to be flagged.
        gradient_flag_value (int, optional): The value to assign to flagged data points.
            Defaults to -9990.
        diff_depth (int, optional): The maximum order of differences to take when calculating the gradient.
            Defaults to 2.
        margin (int, optional): The number of data points to include on either side of the flagged data.
            Defaults to 3.

    Returns:
        pandas.DataFrame: A fully flagged dataset.
    """

    repeat_threshold = int(repeat_threshold)
    diff_depth = int(diff_depth)
    margin = int(margin)

    total_flags = pd.DataFrame(index=data.index, columns=data.columns, dtype=bool)
    total_flags = False

    for forward in [True, False]:
        upper_bound_data = data <= upper_bound
        lower_bound_data = data >= lower_bound

        for diff_num in range(1, diff_depth + 1):
            window_size = (repeat_threshold - 1) * diff_num

            if diff_num == 2:
                diff_data = (
                    diff_data.diff(periods=diff_num)
                    if forward
                    else diff_data.diff(periods=-diff_num)
                )
            else:
                diff_data = (
                    data.diff(periods=diff_num)
                    if forward
                    else data.diff(periods=-diff_num)
                )

            change_data_forward = (abs(diff_data) <= change_threshold).rolling(
                window=window_size
            ).sum() == window_size
            change_data_backward = (abs(diff_data) <= change_threshold).rolling(
                window=window_size
            ).sum().shift(-window_size + 1) == window_size

            change_data = change_data_forward | change_data_backward

            this_flags = change_data & upper_bound_data & lower_bound_data

            total_flags |= this_flags

    # Extend flags based on margin

    extended_flags = total_flags.copy()
    for _ in range(margin):
        extended_flags |= total_flags.shift()
        extended_flags |= total_flags.shift(-1)

    flag_stats = extended_flags.mean().mul(100).round(2).to_dict()

    data[extended_flags] = gradient_flag_value

    return data, flag_stats
