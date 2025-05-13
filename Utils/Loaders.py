"""Functions that load datasets for the UI."""

import os
import sys

import pandas as pd

sys.path.append("..")


if sys.platform.startswith("darwin"):
    LEADING_SLASH = ""
elif sys.platform.startswith("linux"):
    LEADING_SLASH = "/"
else:
    LEADING_SLASH = "./"
PATHNAME_PREFIX = f"{LEADING_SLASH}southernoperations/AIID"


def load_treemap_dataset():
    """Output the treemap dataset."""
    pathname = f"{PATHNAME_PREFIX}/treemap_data.csv"
    if os.path.exists(pathname):
        treemap_data_from_file = pd.read_csv(pathname, parse_dates=[0], index_col=[0])
        cols = [
            x
            for x in treemap_data_from_file.columns
            if not any(
                y in x
                for y in [
                    "Unnamed",
                    "BLADE",
                    "KW",
                    "GEN-SPD-RPM",
                    "WIND-DIR",
                    "YAW-DIR",
                ]
            )
        ]
        treemap_data_from_file = treemap_data_from_file[cols].sort_index()

    else:
        treemap_data_from_file = None
    return treemap_data_from_file


def load_simple_efficiency_dataset():
    """Output the Simple Efficiency dataset."""
    pathname = f"{PATHNAME_PREFIX}/treemap_data_simple_efficiency.csv"
    if os.path.exists(pathname):
        treemap_data_simple_eff = pd.read_csv(pathname, parse_dates=[0], index_col=[0])
        cols = [x for x in treemap_data_simple_eff.columns if "Unnamed" not in x]
        treemap_data_simple_eff = treemap_data_simple_eff[cols].sort_index()
        
    else:
        treemap_data_simple_eff = None
    return treemap_data_simple_eff


def load_yaw_error_dataset():
    """Output the Simple Efficiency dataset."""
    pathname = f"{PATHNAME_PREFIX}/radial_yaw_error.csv"

    if os.path.exists(pathname):
        radial_yaw_error = pd.read_csv(pathname, parse_dates=[0], index_col=[0])
        cols = [x for x in radial_yaw_error.columns if "Unnamed" not in x]
        radial_yaw_error = radial_yaw_error[cols].sort_index()
        
    else:
        radial_yaw_error = None
    return radial_yaw_error


def load_fault_daily_metrics():
    """Output the daily metrics for the fault page.

    The output dataframe looks like this:
    ```
    Date      Turbine  FaultCode  Downtime  FaultCount  LostEnergy    LostRevenue
    8/1/2023  PDK-T001  309       600       1           0.047740897  4.10838578
    8/1/2023  PDK-T002  3273      600       1           0.005271586  0.453651046
    8/1/2023  PDK-T004  309       1200      2           0.009279573  0.798561998
    8/1/2023  PDK-T005  309       1200      2           0.010820179  0.931140271
    8/1/2023  PDK-T008  309       655       1           0.003525356  0.303377707
    8/1/2023  PDK-T009  309       13241     10          0.444306576  38.23520188
    8/1/2023  PDK-T010  309       4546      6           0.180343149  15.51959182
    8/2/2023  PDK-T001  309       1800      1           0.024708218  2.126287914
    8/2/2023  PDK-T002  309       6000      5           0.043500751  3.743496222
    8/2/2023  PDK-T005  309       13800     6           0.210752123  18.13646341
    8/2/2023  PDK-T007  309       10975     5           0.058115785  5.001206166
    8/2/2023  PDK-T008  309       12900     7           0.096346969  8.291225072
    8/2/2023  PDK-T009  309       20926     3           0.522697332  44.9811888
    ```
    About the data set:
    - There will be one record per turbine and fault code per day.
    - Downtime is in seconds.
    - FaultCount is the number of times a new fault event occured regardless of  its length
        (i.e. a fault last 6 hours vs a fault last 10 seconds both get a count of 1).
    - LostEnergy is the lost energy (Expected - Actual Power) under the duration of the fault code. It is in MWh.
    - LostRevenue is directly mapped using the LostEnergy Value. It is in dollars ($).

    """
    pathname = f"{PATHNAME_PREFIX}/daily_turbine_fault.csv"
    if os.path.exists(pathname):
        df_metric = pd.read_csv(
            pathname,
            parse_dates=["Date"],
        )

    else:
        df_metric = None

    return df_metric


def load_fault_metrics():
    """Output the datasets for the pulse and pareto charts."""
    pathname = f"{PATHNAME_PREFIX}/downtime_lost_energy.csv"

    if os.path.exists(pathname):
        df = pd.read_csv(
            pathname,
            parse_dates=[
                "AdjustedStartDateTime",
                "AdjustedEndDateTime",
            ],
        )

        # Add a column of 1s to keep track of the count/number of occurances
        df["Count"] = 1
    else:
        df = None

    return df


def load_fault_code_lookup():
    """Outputs dataset with Fault Codes and Descriptions.

    The output dataframe looks like:
    ```
              Code                     Description Project
    0         0.0                             NaN     BR2
    1         1.0                              OK     BR2
    2         2.0                          ONLINE     BR2
    3         3.0                           RUNUP     BR2
    4         4.0                     MAINTENANCE     BR2
    ...       ...                             ...     ...
    47522  6376.0  Conv/GenWaterCool sec cont err     WHE
    47523  6427.0   IceDetectorSystem Warning Err     WHE
    47524  6470.0      Too low nacel temp ____Â°C     WHE
    47525  6471.0        Transformer Thermo Fault     WHE
    47526  6478.0          Encoder hardware error     WHE
    ```
    """
    pathname = f"{PATHNAME_PREFIX}/fault_description_mapping.csv"
    fault_code_lookup = pd.read_csv(pathname)
    return fault_code_lookup


def load_power_curve_data():
    """Load the power curve dataset."""
    pathname = f"{PATHNAME_PREFIX}/power_curve.csv"
    df = pd.read_csv(pathname, index_col=["Turbine", "Day"])
    df = df.sort_index()
    return df


def load_power_distribution_data():
    """Load the power distribution dataset."""
    pathname = f"{PATHNAME_PREFIX}/power_curve_counts.csv"
    df = pd.read_csv(pathname, index_col=["Turbine", "Day"])
    df = df.sort_index()
    return df


def load_ws_distribution_data():
    """Load file with wind speed distributions by project"""
    df = pd.read_csv(f"assets/data/all_ws_dist.csv", index_col=[0])
    return df


def load_surrogation_strategies():
    """Load the surrogation data."""
    pathname = f"assets/data/surrogation_strategies.csv"
    df = pd.read_csv(pathname)
    df = df.drop_duplicates()
    return df
