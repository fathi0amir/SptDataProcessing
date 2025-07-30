# %%
import numpy as np
import pandas as pd
from multiprocessing import Pool
import math
from functools import partial
import concurrent.futures


data = pd.DataFrame()

for char in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']:
    rng = np.random.default_rng()
    x = rng.lognormal(mean=np.log(10.), sigma=np.log(2), size=1000000)
    y = rng.lognormal(mean=np.log(1.), sigma=np.log(2), size=1000000)

    data = pd.concat(
        [data, pd.DataFrame({'x': x, 'y': y, 'id': char})], 
        ignore_index=True)

def square_them_all(df):
    df = df.copy()
    df['x_sq'] = df['x'] ** 2
    df['y_sq'] = df['y'] ** 2
    return df

def do_them_msd(df):
    msd_results = {}
    df = df.copy()
    max_lag = 50
    for lag in range(1, max_lag + 1):
        dy = df['y'].diff(periods=lag).dropna()
        dx = df['x'].diff(periods=lag).dropna()
        displacement = (dx)**2 + (dy)**2 # convert to microns
        msd_results[lag] = displacement.mean()
    msd_df = pd.DataFrame(list(msd_results.items()), columns=['frame', 'msd'])
    
    df = df.reset_index(drop=True)
    df["MSD"] = msd_df["msd"].reset_index(drop=True)
    df["Lag_T"] = msd_df["frame"].reset_index(drop=True)
    return df

def do_them_msd_pickle(df):
    """
    Calculate the Mean Squared Displacement (MSD) for a single particle trajectory.
    
    Parameters:
    trajectory : np.ndarray, shape (T, d)
        Array of positions with T time steps and d dimensions.
    
    Returns:
    lags : np.ndarray
        Array of lag times.
    msd : np.ndarray
        MSD values for each lag time.
    """
    trajectory = df[['x', 'y']].values
    T, d = trajectory.shape  # T: time steps, d: dimensions
    # lags = np.arange(1, T // 2)  # Lag times (up to half the trajectory length)
    lags = np.arange(1, 51)  # Fixed lag times from 1 to 50
    msd = np.zeros(len(lags))
    
    for idx, lag in enumerate(lags):
        # Calculate squared displacements for this lag time
        displacements = trajectory[lag:] - trajectory[:-lag]
        squared_displacements = np.sum(displacements**2, axis=1)
        # Average over all possible time windows
        msd[idx] = np.mean(squared_displacements)

    df.reset_index(drop=True, inplace=True)
    df['msd'] = np.nan
    df['lag'] = np.nan
    df.loc[:len(lags)-1, 'msd'] = msd
    df.loc[:len(lags)-1, 'lag'] = lags

    return df

def apply_msd_threaded(grouped_df, max_workers=None):
    """Use ThreadPoolExecutor for I/O-bound overhead"""
    groups = [group for name, group in grouped_df]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(do_them_msd, group) for group in groups]
        results = [future.result() for future in futures]
    
    return pd.concat(results, ignore_index=True)


# %%
data2 = data.groupby('id').apply(do_them_msd).reset_index(drop=True)

# %%
data2 = data[data['id'] == 'a'].copy()
data3 = do_them_msd_pickle(data2)


# %%

grouped_df = data.groupby('id')

data6 = apply_msd_threaded(grouped_df, max_workers=None)


# %%




# %%
def split_dataframe_by_uid(df, n_splits=6):
    unique_uids = df['id'].unique()
    n_uids = len(unique_uids)
    
    # Calculate UIDs per split
    uids_per_split = max(1, math.ceil(n_uids / n_splits))
    
    # Create groups of UIDs
    uid_groups = [unique_uids[i:i + uids_per_split] for i in range(0, n_uids, uids_per_split)]
    
    # Ensure we have exactly n_splits groups (pad with empty lists if needed)
    while len(uid_groups) < n_splits:
        uid_groups.append([])
    
    # Create sub-DataFrames
    sub_dfs = []
    for uids in uid_groups[:n_splits]:  # Limit to n_splits
        if len(uids) > 0:
            sub_df = df[df['id'].isin(uids)].copy()
            sub_dfs.append(sub_df)
        else:
            sub_dfs.append(pd.DataFrame(columns=df.columns))  # Empty DataFrame with same structure
    
    return sub_dfs



# Step 2: Define your data processing function
def process_trajectory(sub_df):
    # Your processing logic here
    # Example: Compute distance traveled for each UID
    result = sub_df.groupby('id').apply(do_them_msd)
    return result


# Step 3: Parallel processing function
def parallel_process(df, n_cores=6):
    # Split DataFrame
    sub_dfs = split_dataframe_by_uid(df, n_splits=n_cores)
    
    # Create a pool of workers
    with Pool(processes=n_cores) as pool:
        # Map the processing function to each sub-DataFrame
        results = pool.map(process_trajectory, sub_dfs)
    
    # Combine results
    final_result = pd.concat(results, ignore_index=True)
    return final_result

result = parallel_process(data, n_cores=6)
# %%
import pandas as pd
import math
from concurrent.futures import ProcessPoolExecutor

def split_dataframe_by_uid(df, n_splits=6):
    unique_uids = df['id'].unique()
    n_uids = len(unique_uids)
    uids_per_split = max(1, math.ceil(n_uids / n_splits))
    uid_groups = [unique_uids[i:i + uids_per_split] for i in range(0, n_uids, uids_per_split)]
    while len(uid_groups) < n_splits:
        uid_groups.append([])
    sub_dfs = []
    for uids in uid_groups[:n_splits]:
        if len(uids) > 0:
            sub_df = df[df['id'].isin(uids)].copy()
            sub_dfs.append(sub_df)
        else:
            sub_dfs.append(pd.DataFrame(columns=df.columns))
    return sub_dfs

def process_trajectory(sub_df):
    # Your processing logic here
    return sub_df.groupby('id').apply(do_them_msd)


sub_dfs = split_dataframe_by_uid(data, n_splits=3)
with ProcessPoolExecutor(max_workers=12) as executor:
    results = list(executor.map(process_trajectory, sub_dfs))
final_result = pd.concat(results, ignore_index=True)
print(final_result)
# %%
from joblib import Parallel, delayed

def split_dataframe_by_uid(df, n_splits=6):
    unique_uids = df['id'].unique()
    n_uids = len(unique_uids)
    uids_per_split = max(1, math.ceil(n_uids / n_splits))
    uid_groups = [unique_uids[i:i + uids_per_split] for i in range(0, n_uids, uids_per_split)]
    while len(uid_groups) < n_splits:
        uid_groups.append([])
    sub_dfs = []
    for uids in uid_groups[:n_splits]:
        if len(uids) > 0:
            sub_df = df[df['id'].isin(uids)].copy()
            sub_dfs.append(sub_df)
        else:
            sub_dfs.append(pd.DataFrame(columns=df.columns))
    return sub_dfs

def process_trajectory(sub_df):
    # Your processing logic here
    return sub_df.groupby('id').apply(do_them_msd)


sub_dfs = split_dataframe_by_uid(data, n_splits=24)

results = Parallel(n_jobs=24)(delayed(process_trajectory)(sub_df) for sub_df in sub_dfs)
final_result2 = pd.concat(results, ignore_index=True)
# %%
import concurrent.futures
import pandas as pd

def transform_a(df):
    return df.groupby('category').agg({'value': ['mean', 'std']})

def transform_b(df):
    return df.pivot_table(values='value', index='date', columns='category')

def transform_c(df):
    return df.resample('D', on='date')['value'].sum()

def run_different_transforms(df):
    transforms = [transform_a, transform_b, transform_c]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        # Submit different functions
        futures = [executor.submit(func, df) for func in transforms]
        
        # Get results as they complete
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result(timeout=30)  # 30 second timeout
                results.append(result)
            except Exception as e:
                print(f"Transform failed: {e}")
                results.append(None)
    
    return results