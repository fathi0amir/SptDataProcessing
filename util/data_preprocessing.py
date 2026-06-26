"""
This module provides functions for preprocessing and cleaning TIRF SPT data stored in CSV files.
Functions:
    clean_data(df: pd.DataFrame) -> pd.DataFrame:
        Cleans the input DataFrame by performing various operations such as dropping specific rows,
        selecting and renaming columns, converting data types, and dropping rows with NaN values in 'TRACK_ID'.
    load_csv_files(csv_root: str) -> pd.DataFrame:
        Loads and combines multiple CSV files from a specified directory into a single DataFrame.
"""

from pathlib import Path
import pandas as pd

def clean_data(df)->pd.DataFrame:
    """
    Cleans the input DataFrame by performing the following operations:
    1. Drops the first three rows (assumed to be unit information).
    2. Selects specific columns: **TRACK_ID**, **POSITION_X**, **POSITION_Y**, **POSITION_T**, **FRAME**.
    3. Renames the selected columns to **TrackID**, **X**, **Y**, **T**, and **Frame**.
    4. Converts data types of the selected columns:
        - **TrackID** to int32
        - **X**, **Y**, **T** to float32
        - **Frame** to int32
    5. Drops rows where **TrackID** is NaN.

    Args:
        df (pd.DataFrame): The input DataFrame to be cleaned.
    Returns:
        df (pd.DataFrame): The cleaned DataFrame with **TrackID**, **X**, **Y**, **T**, and **Frame** columns.
    """
    
    # Drop rows 0, 1, and 2 they are unit information
    df = df.drop([0, 1, 2])
    df = df.loc[:, ['TRACK_ID', 'POSITION_X', 'POSITION_Y', 'POSITION_T', 'FRAME']]
    df = df.astype({'TRACK_ID': 'int32', 'POSITION_X': 'float32', 'POSITION_Y': 'float32',
                    'POSITION_T': 'float32', 'FRAME': 'int32'})
    df = df.dropna(subset=['TRACK_ID'])
    df = df.rename(columns={'TRACK_ID': 'TrackID', 'POSITION_X': 'X',
                            'POSITION_Y': 'Y', 'POSITION_T': 'T', 'FRAME': 'Frame'})
    return df

def sort_by_frame(df: pd.DataFrame)->pd.DataFrame:
    """
    Sorts the input DataFrame by 'Frame' in ascending order.
    Args:
        df (pd.DataFrame): The input DataFrame to be sorted.
    Returns:
        pd.DataFrame: The sorted DataFrame.
    """
    df = df.groupby('TrackID', observed=True).apply(lambda x: x.sort_values('Frame')).reset_index(drop=True)
    return df

def format_track_id(x:int)->str:
    """
    Formats an integer as a string with leading zeros to ensure a 
    consistent length of three characters.
    """
    return f'{x:04}'


# Define the path to the directory containing the CSV files
def load_trackmate_csv_files(csv_root)->pd.DataFrame:
    """
    Load and combine multiple CSV files from a specified directory into a single DataFrame.
    The necessary columns are selected, cleaned, and sorted by 'TrackID' and 'Frame'.
    The cleaning steps includes dropping specific rows, renaming columns, converting data types.
    New columns 'FileID' and 'UID' are created to uniquely identify each track. FileID 
    is derived from the filename, and UID is a combination of FileID and TrackID. There is
    also a formatting step for TrackID to ensure it has leading zeros before 
    concatenation with FileID to create UID.
    Args:
        csv_root (str): The root directory containing the CSV files.
    Returns:
        df (pd.DataFrame): A DataFrame containing the combined data from all CSV files in the directory.
        The DataFrame includes columns **TrackID**, **X**, **Y**, **T**, 
        **Frame**, **FileID**, and **UID**.
    """

    path = Path(csv_root)
    csv_files = path.glob('*.csv')

    dfs = []

    for file in csv_files:
        df = pd.read_csv(file, engine='pyarrow')
        df['FileID'] = file.stem
        df = clean_data(df)
        df = df.sort_values(by='TrackID')
        df = sort_by_frame(df)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df['TrackID'] = df['TrackID'].apply(format_track_id)
    df['UID'] = df['FileID'] + '-' + df['TrackID']

    return df 

def clean_trackpy_data(df: pd.DataFrame, exposure_time: float = 0.033)->pd.DataFrame:
    """
    Clean the dataframe from TrackPy to match the format used originally with TrackMate.
    Operations: 
    1. Rename columns: 'particle' to 'TrackID', 'x' to 'X', 'y' to 'Y', and 'frame' to 'Frame'.
    2. Add a new column 'T' 
    3. Sort the dataframe by 'TrackID' and 'Frame'.
    4. Convert TrackID to string with leading zeros to ensure a consistent length of three characters.
    

    Args: 
        df (pd.DataFrame): The input DataFrame to be cleaned.
        exposure_time (float): The exposure time in seconds. Default is 0.033 seconds.
    Returns:
        pd.DataFrame: The cleaned DataFrame
    """
    df = df.rename(columns={'particle': 'TrackID', 'x': 'X', 'y': 'Y', 'frame': 'Frame'})
    df['T'] = df['Frame'].astype('float32') * exposure_time
    df = df.sort_values(by=['TrackID', 'Frame']).reset_index(drop=True)
    df['TrackID'] = df['TrackID'].apply(lambda x: f"{x:03d}")

    return df

def load_trackpy_parquet(data_wd: Path, exposure_time: float = 0.033, suffix: str = None)->pd.DataFrame:

    """
    Load and combine multiple Parquet files from a specified directory into a single DataFrame.

    The data from TrackPy is in pixels, so the X and Y coordinates are multiplied by 65 to convert them to nanometers.
    """
    path = Path(data_wd)
    parquet_files = path.glob('*.parquet')
    dfs = []

    for file in parquet_files:
        df = pd.read_parquet(file, engine='fastparquet')
        if suffix is not None and suffix in file.stem:
            fileid = file.stem.replace(suffix, '')
        else:
            fileid = file.stem
        df['FileID'] = fileid
        df = clean_trackpy_data(df, exposure_time=exposure_time)
        df['UID'] = df['FileID'] + '-' + df['TrackID']
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df[['X', 'Y']] *= 65

    return df
