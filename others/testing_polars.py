# %%

import numpy as np
import polars as pl
import math
from concurrent.futures import ProcessPoolExecutor

data = pl.DataFrame()

for char in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']:
    rng = np.random.default_rng()
    x = rng.lognormal(mean=np.log(10.), sigma=np.log(2), size=1000000)
    y = rng.lognormal(mean=np.log(1.), sigma=np.log(2), size=1000000)

    data = pl.concat(
        [data, pl.DataFrame({'x': x, 'y': y, 'id': char})], 
        how='vertical')

def square_them_all(df):
    df = df.copy()
    df['x_sq'] = df['x'] ** 2
    df['y_sq'] = df['y'] ** 2
    return df

def do_them_msd(df):
    msd_results = {}
    # df = df.copy()
    max_lag = 50
    for lag in range(1, max_lag + 1):
        dy = df['y'].diff()
        dx = df['x'].diff()
        displacement = (dx)**2 + (dy)**2 # convert to microns
        msd_results[lag] = displacement.mean()
    msd_df = pl.DataFrame(list(msd_results.items()))

    df = df.reset_index(drop=True)
    df["MSD"] = msd_df["msd"]
    df["Lag_T"] = msd_df["frame"]
    return df



# %%
data2 = data.group_by('id').map_groups(do_them_msd, strict=False)


# %%

def split_dataframe_by_uid(df, n_splits=6):
    unique_uids = df['id'].unique().to_numpy()
    n_uids = len(unique_uids)
    uids_per_split = max(1, math.ceil(n_uids / n_splits))
    uid_groups = [unique_uids[i:i + uids_per_split] for i in range(0, n_uids, uids_per_split)]
    while len(uid_groups) < n_splits:
        uid_groups.append([])
    sub_dfs = []
    for uids in uid_groups[:n_splits]:
        if len(uids) > 0:
            sub_df = df.filter(pl.col('id').is_in(uids))
            sub_dfs.append(sub_df)
        else:
            sub_dfs.append(pl.DataFrame(schema=df.schema))
    return sub_dfs


def process_trajectory(sub_df):
    # Group by 'id' and apply do_them_msd
    results = []
    for uid, group_df in sub_df.groupby('id'):
        results.append(do_them_msd(group_df))
    return pl.concat(results)


sub_dfs = split_dataframe_by_uid(data, n_splits=3)
with ProcessPoolExecutor(max_workers=12) as executor:
    results = list(executor.map(process_trajectory, sub_dfs))
final_result = pl.concat(results)
print(final_result)


# %%
# Sample DataFrame
df = pl.DataFrame({
    "group": ["A", "A", "B", "B", "A"],
    "value1": [1, 2, 3, 4, 5],
    "value2": [10, 20, 30, 40, 50]
})

# Define a custom function that takes and returns a DataFrame
# This example function normalizes 'value1' within each group
def normalize_by_group(group_df: pl.DataFrame) -> pl.DataFrame:
    max_val = group_df["value1"].max()
    return group_df.with_columns(
        (pl.col("value1") / max_val).alias("normalized_value1")
    )

# Apply the custom function
# Note: The result is a single DataFrame with the combined results
result_apply = df.group_by("group", maintain_order=True).map_groups(normalize_by_group)

print(result_apply)
# %%
