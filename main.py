# %% # MARK: Imports
from pathlib import Path
from re import sub
import numpy as np
import pandas as pd # noqa
import plotly.express as px
import importlib
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import (
    datashade,
    dynspread,
    rasterize,
    shade,
    spread,
) 

# Custom modules
import util.data_preprocessing as dpp
import util.analysis_functions as nlss
import util.look_and_feel as laf
import util.constants as const

hv.extension('bokeh') #type: ignore

# * ====================================
# * Reload modules
# Reload modules to ensure the latest changes are applied
def reload_modules():
    """Reloads the custom modules to ensure the latest changes are applied.
    """
    importlib.reload(dpp)
    importlib.reload(nlss)
    importlib.reload(laf)
    importlib.reload(const)

# Call the reload_modules() function to ensure all modules are up-to-date
reload_modules()
#
#
#
# %% # * ====================================
# MARK: Load data
# Load the CSV files from the specified directory
data_path = Path(r"C:\Users\amir0\Desktop\GD_DOPC_20250825\TrackingResults")
df = dpp.load_csv_files(data_path)

df.info()
df.head(20)
#
#
#
# %% # * ====================================
# * Filter stationary tracks
# ! I need to fix this part
def calculate_top_distance_sum(df: pd.DataFrame):
    """
    Calculate the distance from the origin for each point in the DataFrame.
    Maximum jump distance should be greater than localization error.
    """
    dx = df['X'].diff().dropna()
    dy = df['Y'].diff().dropna()
    dist = np.sqrt(dx**2 + dy**2)
    dist_top_10 = dist.nlargest(10)
    return dist_top_10.sum()

df = df.groupby('UID').filter(lambda g: calculate_top_distance_sum(g) > 10* 40.0)
#
#
#
#
# %% # * ====================================
# MARK: Save Data
df.to_parquet(data_path / 'tracking_results.parquet', index=False)
# Save the data when you are done. 
# Parquet format is more efficient for large datasets and maintains data types.
# Otherwise one can use df.to_csv("<path_to_save_csv>") to save the data in CSV format.
#
#
#
# * ====================================
# MARK: Data Preprocessing
# All the analysis are done in this section
# There are more ananlysis functions in util/analysis_functions.py
# such as typical anomalous diffusion analysis, and JD analysis with 
# more than two components in an anomalous diffusion model
# %% # * Add the MSD Lag_T columns
# This is necessary for all the MSD based analysis
df = df.groupby('UID').apply(nlss.calculate_msd).reset_index(drop=True)
# Normalize MSD by the first value
df['MSD_norm'] = df.groupby('UID')['MSD'].transform(lambda x: x / x.iloc[0])
#
#
# %% # * Calculate D in normal diffusion model
df = df.groupby('UID').apply(nlss.calculate_diff_d).reset_index(drop=True)
#
#
#
# %% # * Fit Norm MSD in log-log scale to find alpha
# Use the alpha value to flag each trajectory as well. 
# The flags are based on ALPHA_THRESHOLDS defined in util/constants.py
df = df.groupby('UID').apply(nlss.flag_alpha_by_fit).reset_index(drop=True)
#
# * Calculate D when Alpha is fixed (exact value for that trajectory)
df = df.groupby('UID').apply(nlss.calc_d_fix_alpha).reset_index(drop=True)
#
#
#
#
#
# %% # * Calculate the mean of Alpha for each alpha flag (class)
df['Alpha_Mean'] = df.groupby('Alpha_Flag_Fit')['Alpha'].transform('mean')
#
#
#
# %% # * Calculate D when Alpha is fixed to the mean of its class
df = df.groupby('UID').apply(nlss.calc_d_mean_alpha).reset_index(drop=True)
#
#
#

# %% # * Create a new dataframe with two columns: Alpha Flag , mean Alpha for each flag (class)
alphas = df.groupby('Alpha_Flag_Fit')['Alpha'].mean()
print("Alpha Mean for each Alpha Flag:")
print(alphas)
#
#
#
# %% # * Calculate the JD counts and bin centers
df = df.groupby('UID').apply(nlss.calculate_jd).reset_index(drop=True)
#
#
#
# %% # * Perform JD fitting with one component
df = df.groupby('UID').apply(nlss.fit_jd_1exp_norm).reset_index(drop=True)
#
#
#
# %% # * Perform JD fitting with two components
df = df.groupby('UID').apply(nlss.fit_jd_2exp_norm).reset_index(drop=True)
#
#
#
# %% # * Get the dataframe info and head
df.info()
df.head()
#
#
#

# %% # * ====================================
# * Stats on Alpha Flags
# Calculate the percentage of trajectories in each Alpha_Flag_Fit category

flag_percentages = nlss.calculate_flag_percentages(df)

print("Alpha_Flag_Fit statistics (percentage of trajectories):")
for flag, pct in flag_percentages.items():
    print(f"{flag}: {pct}%")
# %% # * ====================================
# Plot a histogram of the 'D' column
reload_modules()
# laf.plotly_plot_diff_coef_hist(df, column='D_Fixed_Alpha')
# laf.plotly_plot_diff_coef_logloghist(df)
laf.plotly_plot_diff_coef_loglogarea(df, column='D_Fixed_Alpha')
#
#
#
# %% # * ====================================
# Plot a histogram of the 'Alpha' column
reload_modules()
laf.plotly_plot_alpha_hist(df)
#
#
#
# %% # * ====================================
# Plot MSD vs Lag_T for each FileID and TrackID
# MSDs are normalized by the first MSD value for each UID
# This is to compare the MSDs across all the trajectories
reload_modules()
# fig = laf.plotly_plot_norm_loglog_msd(df)
# fig.show(renderer="svg")
#
# laf.vega_plot_msd_loglog_fast(df, bin_size=0.03)
laf.holoviz_plot_msd_loglog_fast(df, bin_num=150)
#
#
#
# %% # * ====================================
# Plot MSD vs Lag_T for each FileID and TrackID
# Color from the alpha flag
reload_modules()
fig = laf.plotly_plot_norm_msd_grouped(df, alphas)
fig.show(renderer="svg")
#
#
#
# %% # * ====================================
# Plot a scatter plot of D_Fixed_Alpha vs Alpha
reload_modules()
laf.plotly_plot_diff_coef_vs_alpha(df)
#
#
#
# %% # * ====================================
# filter and plot for normal trajectories based on the alpha flag
#
reload_modules()
normal_trajectories = df[df['Alpha_Flag_Fit'] == 'normal']
fig = px.line(normal_trajectories, x='X', y='Y', color='UID')
fig = laf.plotly_style_tracks(fig)
laf.set_plotly_config(fig, width=600, height=600)  # wrapper for fig.show(config=config)
#
#

#%% # * ====================================
# Plot a single random normal trajectory
#
reload_modules()
random_trajectory = normal_trajectories[
    normal_trajectories['UID'] == normal_trajectories['UID'].unique()[10]]
fig = px.line(random_trajectory, x='X', y='Y')
fig = laf.plotly_style_single_track(fig)
laf.set_plotly_config(fig, width=600, height=600)  # wrapper for fig.show(config=config)
#
#
#
# %% # * ====================================
# Filter and plot for subdiffusive trajectories
subdiffusive_trajectories = df[df['Alpha_Flag_Fit'] == 'sub']
fig = px.line(subdiffusive_trajectories, x='X', y='Y', color='UID')
fig = laf.plotly_style_tracks(fig)
laf.set_plotly_config(fig, width=600, height=600)  # wrapper for fig.show(config=config)
#
#
#
# %% # * ====================================
# Plot a random subdiffusive trajectory
random_subdiffusive_trajectory = subdiffusive_trajectories[
    subdiffusive_trajectories['UID'] == subdiffusive_trajectories['UID'].unique()[10]]
fig = px.line(random_subdiffusive_trajectory, x='X', y='Y')
fig = laf.plotly_style_single_track(fig)
laf.set_plotly_config(fig, width=600, height=600)  # wrapper for fig.show(config=config)
#
#
#
# %% # * ====================================
# Filter and plot for superdiffusive trajectories
superdiffusive_trajectories = df[df['Alpha_Flag_Fit'] == 'sup']
fig = px.line(superdiffusive_trajectories, x='X', y='Y', color='UID')
fig = laf.plotly_style_tracks(fig)
laf.set_plotly_config(fig, width=600, height=600)  # wrapper for fig.show(config=config)
#
#
#
# %% # * ====================================
# Plot a random superdiffusive trajectory
random_superdiffusive_trajectory = superdiffusive_trajectories[
    superdiffusive_trajectories['UID'] == superdiffusive_trajectories['UID'].unique()[0]]
fig = px.line(random_superdiffusive_trajectory, x='X', y='Y')
fig = laf.plotly_style_single_track(fig)
laf.set_plotly_config(fig, width=600, height=600)  # wrapper for fig.show(config=config)
#
#
#
# %% # * ====================================
# Filter and plot for ignored trajectories
ignore_trajectories = df[df['Alpha_Flag_Fit'] == 'ignore']
fig = px.line(ignore_trajectories, x='X', y='Y', color='UID')
fig = laf.plotly_style_tracks(fig)
laf.set_plotly_config(fig, width=600, height=600)  # wrapper for fig.show(config=config)
#
#
#
# %% # * ====================================
# Plot a random ignored trajectory
random_ignored_trajectory = ignore_trajectories[
    ignore_trajectories['UID'] == ignore_trajectories['UID'].unique()[10]]
fig = px.line(random_ignored_trajectory, x='X', y='Y')
fig = laf.plotly_style_single_track(fig)
laf.set_plotly_config(fig, width=600, height=600)  # wrapper for fig.show(config=config)
#
#
#
# %%
# Plot for all alpha flags except 'ignore' 
# Filter for ignored trajectories
ignore_trajectories = df[df['Alpha_Flag_Fit'] != 'ignore']

fig = px.line()
for id, g in ignore_trajectories.groupby('UID'):
    c='#00CC96'
    if g['Alpha_Flag_Fit'].iloc[0] == 'sup':
        c = '#EF553B'
    elif g['Alpha_Flag_Fit'].iloc[0] == 'sub':
        c = '#636EFA'
    fig.add_scatter(x=g['X'], y=g['Y'], mode='lines',
        line=dict(width=1, color=c), name=f'UID: {id}')

fig.update_traces(
    hovertemplate=
        'X: %{x}<br>' +
        'Y: %{y}<br>'
)
fig = laf.plotly_style_tracks(fig)
laf.set_plotly_config(fig, width=600, height=600)  # wrapper for fig.show(config=config)
#
#
#
# %% # * ====================================
# * =========================================
# * =========================================
# MARK: Plot MSD JD
# Create a Holoviews Dataset for the MSD vs Lag_T plot
# Overlay all curves for each TrackID
msd_overlay = hv.NdOverlay({
    uid: hv.Curve(
        (group['Lag_T'].iloc[0:6],  # Use first 6 points for MSD
        group['MSD_norm'].iloc[0:6]),  # Normalize by first value
        label=str(uid)
    ).opts(
        line_width=2, color='blue', alpha=0.05
    )
    for uid, group in df.groupby('TrackID')
})

msd_overlay.opts(
    title="Normalized MSD vs Lag_T for all TrackIDs",
    xlabel="Lag Time (s)",
    ylabel="Normalized Mean Squared Displacement (MSD)",
    logx=True,
    logy=True,
    xlim=(0.03, 0.2),
    ylim=(0.9, 10),
    show_legend=False,
    width=600,
    height=400,
    toolbar='above',
    backend_opts={"plot.output_backend": "svg"}
)

# Display the plot
msd_overlay #type: ignore
#
#
#
# %%
# hv.Path(df, kdims=['Lag_T', 'MSD'], groupby='UID').opts(alpha=0.1)
#

# Create a Holoviews NdOverlay for the MSD vs Lag_T plot grouped by UID
msd_overlay = hv.NdOverlay({
    uid: hv.Curve(
        (group['Lag_T'].iloc[0:6], 
        group['MSD'].iloc[0:6] / group['MSD'].iloc[0]),  # Normalize by first value
        label=str(uid)
    ).opts()
    for uid, group in df.groupby('UID')
}).opts() #type: ignore

# Display the plot
# spread(rasterize(paths), px=1) = rasterize(paths, line_width=2)
datashade(msd_overlay, line_width=1).opts( #type: ignore
    alpha=1,
    title="MSD vs Lag_T for each UID",
    xlabel="Lag Time (s)",
    ylabel="Mean Squared Displacement (MSD)", logy=True, logx=True,
    # xlim=(0.03, 0.2),
    # ylim=(0.9, 10),
    width=600,
    height=400,
    toolbar='above',
    # clim=(0, 40),
    # backend_opts={"plot.output_backend": "svg"}
    ) #type: ignore
# msd_overlay


# %% # * ====================================
# filter the trajectories based on the diffusion coefficient
# Filter the data for a specific FileID
file_ids = df['FileID'].unique()
filtered_df = df[df['FileID'] == file_ids[0]]
filtered_df = filtered_df[filtered_df['D_Fixed_Alpha'] > 0.09]
# filtered_df = filtered_df[filtered_df['D'] > 0.7]

# Plot the filtered trajectories
fig = px.line(filtered_df, x='X', y='Y', color='TrackID')
fig = laf.plotly_style_tracks(fig)
config = {'toImageButtonOptions': {'scale': 4}}
fig.show(config=config)
#
#
#

# %% # * ====================================
# Plot JD_Freq against JD_bin_centers for one of the UID in the df
uid_to_plot = df['UID'].unique()[4]
jd_data = df[df['UID'] == uid_to_plot]
x= jd_data['JD_Bin_Center']
y = jd_data['JD_Freq']
d = jd_data['JD1x_D'].iloc[0]
alpha = jd_data['JD1x_Alpha'].iloc[0]
fig = px.line(jd_data, x=x, y=y, title=f'JD_Freq vs JD_bin_centers for UID {uid_to_plot}')
fig.add_scatter(x=x, y=nlss.jd_1exp(x, d, alpha), mode='lines', name='JD Fit 1exp')
fig.update_layout(
    xaxis_title='JD Bin Centers',
    yaxis_title='JD Frequency',
    paper_bgcolor='rgb(255, 255, 255)',
    plot_bgcolor='rgb(255, 255, 255)'
)
fig.show()
#
#
#
# %% # * ====================================
# Plot a histogram of the 'D' column
grouped_df = df.groupby('UID')['JD1xn_D'].first().reset_index()
grouped_df = grouped_df[(grouped_df['JD1xn_D'] < 1.5) & (grouped_df['JD1xn_D'] > 0.1)]
fig = px.histogram(x=grouped_df['JD1xn_D'], nbins=50)
fig.update_layout(
    xaxis_title='Diffusion Coefficient (D)',
    yaxis_title='Count',
    bargap=0.01,
    paper_bgcolor='rgba(255, 255, 255, 0.90)',
    plot_bgcolor='rgba(60, 60, 60, 0.44)'
)
fig.show()

# %% # * ====================================
# Plot a histogram of the '𝛂' column
grouped_df = df.groupby('UID')['JD1x_Alpha'].first().reset_index()
grouped_df = grouped_df[(grouped_df['JD1x_D'] < 1.1) & (grouped_df['JD1x_D'] > 0.1)]
fig = px.histogram(x=grouped_df['JD1x_Alpha'], nbins=100)
fig.update_layout(
    xaxis_title='Diffusion Coefficient (D)',
    yaxis_title='Count',
    bargap=0.01,
    paper_bgcolor='rgba(255, 255, 255, 0.90)',
    plot_bgcolor='rgba(60, 60, 60, 0.44)'
)
fig.show()
#
#
#
# %% # * ====================================
# Plot JD_Freq against JD_bin_centers for one of the UID in the df
uid_to_plot = df['UID'].unique()[15]
jd_data = df[df['UID'] == uid_to_plot]
x= jd_data['JD_Bin_Center']
y = jd_data['JD_Freq']
d = jd_data['JD1xn_D'].iloc[0]
fig = px.line(jd_data, x=x, y=y, title=f'JD_Freq vs JD_bin_centers for UID {uid_to_plot}')
fig.add_scatter(x=x, y=nlss.jd_1exp_norm(x, d), mode='lines', name='JD Fit 1exp')
fig.update_layout(
    xaxis_title='JD Bin Centers',
    yaxis_title='JD Frequency',
    paper_bgcolor='rgb(255, 255, 255)',
    plot_bgcolor='rgb(255, 255, 255)'
)

fig.show()
#
#
#
# %% # * ====================================
# Plot JD_Freq against JD_bin_centers for one of the UID in the df
uid_to_plot = df['UID'].unique()[15]
jd_data = df[df['UID'] == uid_to_plot]

x= jd_data['JD_Bin_Center']
y = jd_data['JD_Freq']
a1 = jd_data['JD2xn_a1'].iloc[0]
a2 = jd_data['JD2xn_a2'].iloc[0]
d1 = jd_data['JD2xn_D1'].iloc[0]
d2 = jd_data['JD2xn_D2'].iloc[0]

fig = px.line(jd_data, x=x, y=y, title=f'JD_Freq vs JD_bin_centers for UID {uid_to_plot}')
fig.add_scatter(x=x, y=nlss.jd_2exp_norm(x, a1, a2, d1, d2), mode='lines', name='JD Fit 1exp')
fig.update_layout(
    xaxis_title='JD Bin Centers',
    yaxis_title='JD Frequency',
    paper_bgcolor='rgb(255, 255, 255)',
    plot_bgcolor='rgb(255, 255, 255)'
)

fig.show()
#
#
#
# %% # * ====================================
# Plot JD_Freq against JD_bin_centers for one of the UID in the df
uid_to_plot = df['UID'].unique()[300]
jd_data = df[df['UID'] == uid_to_plot]

x= jd_data['JD_Bin_Center']
y = jd_data['JD_Freq']
a1 = jd_data['JD2x_a1'].iloc[0]
a2 = jd_data['JD2x_a2'].iloc[0]
d1 = jd_data['JD2x_D1'].iloc[0]
d2 = jd_data['JD2x_D2'].iloc[0]
c1 = jd_data['JD2x_Alpha1'].iloc[0]
c2 = jd_data['JD2x_Alpha2'].iloc[0]

fig = px.line(jd_data, x=x, y=y, title=f'JD_Freq vs JD_bin_centers for UID {uid_to_plot}')
fig.add_scatter(x=x, y=nlss.jd_2exp(x, a1, a2, d1, d2, c1, c2), mode='lines', name='JD Fit 1exp')
fig.update_layout(
    xaxis_title='JD Bin Centers',
    yaxis_title='JD Frequency',
    paper_bgcolor='rgb(255, 255, 255)',
    plot_bgcolor='rgb(255, 255, 255)'
)

fig.show()
#
#
#
# %%# * ====================================
# Plot a histogram of the 'D' column
grouped_df = df.groupby('UID')['JD2xn_D2'].first().reset_index()
grouped_df = grouped_df[(grouped_df['JD2xn_D2'] < 2) & (grouped_df['JD2xn_D2'] > 0.1)]
fig = px.histogram(x=grouped_df['JD2xn_D2'], nbins=50)
fig.update_layout(
    xaxis_title='Diffusion Coefficient (D)',
    yaxis_title='Count',
    bargap=0.01,
    paper_bgcolor='rgba(255, 255, 255, 0.90)',
    plot_bgcolor='rgba(60, 60, 60, 0.44)'
)
fig.show()

# %% # * ====================================
# Plot a histogram of the 'D' column
grouped_df = df.groupby('UID')['JD2x_Alpha2'].first().reset_index()
grouped_df = grouped_df[(grouped_df['JD2x_Alpha2'] < 2) & (grouped_df['JD2x_Alpha2'] > 0)]
fig = px.histogram(x=grouped_df['JD2x_Alpha2'], nbins=100)
fig.update_layout(
    xaxis_title='Diffusion Coefficient (D)',
    yaxis_title='Count',
    bargap=0.01,
    paper_bgcolor='rgba(255, 255, 255, 0.90)',
    plot_bgcolor='rgba(60, 60, 60, 0.44)'
)
fig.show()
#
#
#
# %% # * ====================================
# Plot a histogram of the 'D' column
column_name = 'D_Fixed_Alpha'
grouped_df = df.groupby('UID')[column_name].first().reset_index()
grouped_df = grouped_df[(grouped_df[column_name] < 5) & (grouped_df[column_name] > 0.07)]
fig = px.histogram(x=grouped_df[column_name], nbins=50)
fig.update_layout(
    xaxis_title='Diffusion Coefficient µm²/s',
    yaxis_title='Count',
    bargap=0.01,
    paper_bgcolor='rgb(255, 255, 255)',
    plot_bgcolor='rgb(220, 220, 220)',
    # xaxis_type='log'
)
fig.show()
#
#
#

#%% # MARK: Find track
# * ====================================
# * Plot all tracks, to find a long track
# Filter the data for a specific UID
good_length = 100
filtered_df = df.groupby('UID').filter(lambda x: len(x) > good_length)
# Convert the series into a DataFrame
filtered_df = filtered_df.reset_index(drop=True)

fig = px.line(filtered_df, x='X', y='Y', color='UID')
fig = laf.plotly_style_tracks(fig)
laf.set_plotly_config(fig) # wrapper for fig.show(config=config)
#
#
#
# %% # MARK: Select track
# * ====================================
# * Load the 'good' track for further analysis
# track = filtered_df[filtered_df['UID'] == 'BMP-TAT-S001-60min-ROI02-1146']
track = filtered_df[filtered_df['UID'] == filtered_df['UID'].unique()[0]]
fig = px.line(track, x='X', y='Y', color='UID')
fig = laf.plotly_style_tracks(fig)
laf.set_plotly_config(fig) # wrapper for fig.show(config=config)
#
#
#
# %% # MARK: Confinement 
# * ====================================
# * Calculate the confinement level for the track
track = nlss.calc_confinement_level(track)
# track.info()
# Columns of interest: MW_D, Conf_Level, Frame
fig = px.line(track, x='Frame', y='Conf_Level', 
                title=f'Confinement Level vs Frame {const.WINDOW_SIZE} & {const.MSD_FIT_POINTS}')
fig.update_layout(
    template='plotly_white', 
    xaxis=dict(
        title='Frame',
        showline=True,      # Show axis line
        showgrid=True,      # Show grid lines
        showticklabels=True,
        linecolor='black',
        ticks='inside',
        mirror=True,        # Mirror the ticks on the opposite side
        linewidth=2,        # Width of the axis line
    ),
    yaxis=dict(
        title='Confinement Level',
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='black',
        ticks='inside',
        mirror=True,
        linewidth=2,
    )
)
laf.set_plotly_config(fig) # wrapper for fig.show(config=config)
#
#
#
# %% # MARK: Interactive plots
# # * ====================================
# * Interactive plots with Holoviews
hvds = hv.Dataset(track)

# Create Holoviews objects for the plots
track_plot = hv.Points(hvds, ["X", "Y"]).opts(
    title="Track Plot", xlabel="X", ylabel="Y"
)
track_plot.opts( #type: ignore
    backend_opts={"plot.output_backend": "svg"}
)

conf_level_plot = hv.Scatter(hvds, "Frame", "Conf_Level").opts(
    title="Confinement Level vs Frame",
    xlabel="Frame",
    ylabel="Confinement Level"
)
conf_level_plot.opts( #type: ignore
    backend_opts={"plot.output_backend": "svg"}
)
layout = (track_plot + conf_level_plot) #type: ignore
ls = hv.link_selections.instance()
# Link the selections
ls(layout, #type: ignore
    selected_color='#fc4a4a', 
    unselected_alpha=1, 
    unselected_color='#5a9d5a'
)
#
#
#
# %% # MARK: Tr_MSD
# * ====================================
# * Calculate the Transient MSD

reload_modules()  # Reload modules to ensure the latest changes are applied

track = nlss.calculate_diff_d_moving_window(track)
# track.info()
# Columns of interest: MW_D, Conf_Level, Frame
fig = px.line(
    track,
    x="Frame",
    y="MW_D",
    title=f"Transient MSD vs Frame {const.TMSD_WINDOW_SIZE} & {const.TMSD_FIT_POINTS}",
)
fig.update_layout(
    xaxis_title="Frame",
    yaxis_title="Transient MSD",
    # paper_bgcolor="rgb(255, 255, 255)",
    # plot_bgcolor="rgb(220, 220, 220)",
    template="presentation"
)
laf.set_plotly_config(fig) # wrapper for fig.show(config=config)
#
#
#

# %% # MARK: Interactive plots
# # * ====================================
# * Interactive plots with Holoviews
# Her we plot transient D against Frame
hvds = hv.Dataset(track)
svgopts = opts(backend_opts={"plot.output_backend": "svg"})

# Create Holoviews objects for the plots
track_plot = hv.Points(hvds, ["X", "Y"]).opts(
    title="Track Plot", xlabel="X", ylabel="Y"
)
track_plot.opts(svgopts) #type: ignore

conf_level_plot = hv.Scatter(hvds, "Frame", "MW_D").opts(
    title="Confinement Level vs Frame",
    xlabel="Frame",
    ylabel="Confinement Level"
)
conf_level_plot.opts(svgopts) #type: ignore

layout = (track_plot + conf_level_plot) #type: ignore
ls = hv.link_selections.instance()
# Link the selections
ls(layout, #type: ignore
    selected_color='#fc4a4a', 
    unselected_alpha=1, 
    unselected_color='#5a9d5a'
)
#
#
#

# %%
import altair as alt
from vega_datasets import data
import vegafusion #noqa

# Enable VegaFusion for server-side transforms
alt.data_transformers.enable("vegafusion")

selection = alt.selection_interval(encodings=["x", "y"])

# Scatter plot with consistent color by Origin, using opacity for selection
scatter = (
    alt.Chart(track.reset_index())
    .mark_point()
    .encode(
        x=alt.X(
            "X", title="X", scale=alt.Scale(domain=[track["X"].min(), track["X"].max()])
        ),
        y=alt.Y(
            "Y", title="Y", scale=alt.Scale(domain=[track["Y"].min(), track["Y"].max()])
        ),
        opacity=alt.condition(selection, alt.value(1.0), alt.value(0.1)),
    )
    .interactive()
    .add_params(selection)
    .properties(
        width=400,
        height=300,
    )
)

# Line plot of Conf_Level against Frame
line_plot = (
    alt.Chart(track.reset_index())
    .mark_line()
    .encode(
        x=alt.X(
            "Frame",
            title="Frame",
            scale=alt.Scale(domain=[track["Frame"].min(), track["Frame"].max()]),
        ),
        y=alt.Y(
            "Conf_Level",
            title="Confinement Level",
            scale=alt.Scale(
                domain=[track["Conf_Level"].min(), track["Conf_Level"].max()]
            ),
        ),
        tooltip=["Frame", "Conf_Level"],
    )
    .interactive()
    .transform_filter(selection)
    .properties(width=400, height=300)
)

# Combine scatter and line plot vertically
chart = scatter & line_plot
chart.show()
#
#
#
# %%
import altair as alt
from vega_datasets import data
import vegafusion #noqa

# Enable VegaFusion for server-side transforms
alt.data_transformers.enable("vegafusion")
alt.renderers.enable("jupyter")
# alt.renderers.enable("svg")



# Altair plot: MSD vs Lag_T, all lines same color
msd_chart = (
    alt.Chart(df.reset_index())
    .mark_line(color='blue', opacity=0.02, clip=True)
    .encode(
        x=alt.X(
            "Lag_T:Q",
            title="Lag Time (s)",
            scale=alt.Scale(type="log", domain=[0.03, 0.2])
        ),
        y=alt.Y(
            "MSD_norm:Q",
            title="Mean Squared Displacement (MSD)",
            scale=alt.Scale(type="log", domain=[1, 10])
        ),
        detail="UID:O",  # Group lines by UID, but do not color by UID
    )
    .properties(
        width=600,
        height=400,
        title="MSD vs Lag_T (log-log scale)"
    )
)
msd_chart.show()



# %%
alt.Chart(df.reset_index()).mark_rect(clip=True).encode(
    x=alt.X('Lag_T:Q').bin(maxbins=250).scale(type='log'),
    y=alt.Y('MSD_norm:Q').bin(maxbins=250).scale(type='log'),
    color=alt.Color('count():Q').scale(scheme='greenblue', domain=[0, 500])
)



# %%
alt.Chart(df.reset_index()).mark_rect(clip=False).encode(
    x=alt.X('Lag_T:Q').bin(step=0.02).scale(type='log'),
    y=alt.Y('MSD_norm:Q').bin(step=0.02).scale(type='log'),
    color=alt.Color('count():Q').scale(scheme='greenblue')
)
# %% # MARK: Ensemble MSD
enmsd = nlss.calculate_ensemble_msd(df)
# %%
fig = px.line()
fig.add_scatter(
    x=enmsd['Lag_T'],
    y=enmsd['EnMSD'],
    mode='lines',
    name='Ensemble MSD',
    line=dict(color='red', width=2)
)

fig.update_layout(
    template='plotly_white',
    xaxis_title='Lag Time (s)',
    yaxis_title='Ensemble Mean Squared Displacement (MSD)',
    title='Ensemble MSD vs Lag_T for all FileIDs and TrackIDs',
    width=800,
    height=600,
    # xaxis_range=[0.01, 0.2],
    # yaxis_range=[np.log10(0.01), None],
    xaxis_type='log',
    yaxis_type='log',
    showlegend=False,
    xaxis=dict(
        showline=True,
        linecolor='black',
        linewidth=2,
        mirror=True  # Draws axis lines on both bottom/top or left/right
    ),
    yaxis=dict(
        showline=True,
        linecolor='black',
        linewidth=2,
        mirror=True
    ),
)
# %%
import altair as alt
from vega_datasets import data

source = data.seattle_weather()

alt.Chart(source).mark_rect().encode(
    alt.X("date(date):O").axis(labelAngle=0, format="%e").title("Day"),
    alt.Y("month(date):O").title("Month"),
    alt.Color("max(temp_max):Q").title("Max Temp"),
)
# %%
import altair as alt
import vegafusion  # noqa: F401
import pandas as pd
import numpy as np

# alt.data_transformers.enable("vegafusion")
alt.data_transformers.disable_max_rows()

x = np.random.lognormal(mean=np.log10(2.), sigma=1, size=100000)
y = np.random.lognormal(mean=np.log10(2.), sigma=1, size=100000)
df = pd.DataFrame({'x': x, 'y': y})

chart = alt.Chart(df).mark_rect(clip=True).encode(
    x=alt.X('x:Q', bin=alt.Bin(maxbins=500), scale=alt.Scale(type='log', domain=[0.1, 10])),
    y=alt.Y('y:Q', bin=alt.Bin(maxbins=500), scale=alt.Scale(type='log', domain=[0.1, 10])),
    color=alt.Color('count():Q', scale=alt.Scale(scheme='greenblue'))
).properties(
    width=400,
    height=400,
    title='Heatmap of Log-Normal Distribution'
)

chart.show()
# %%

import plotly.express as px
import pandas as pd
import numpy as np

x = np.random.lognormal(mean=np.log10(2.), sigma=1, size=100000)
y = np.random.lognormal(mean=np.log10(2.), sigma=1, size=100000)
df = pd.DataFrame({'x': x, 'y': y})

# Filter out any zero or negative values to avoid log scale issues
df_filtered = df[(df['x'] > 0) & (df['y'] > 0)]

fig = px.density_heatmap(
    df_filtered,
    x='x',
    y='y',
    nbinsx=1000,
    nbinsy=1000,
    log_x=True,
    log_y=True,
)
fig.update_layout(
    width=600,
    height=600,
    xaxis_range=[-1, 1],  # log10(0.1)= -1, log10(10)=1
    yaxis_range=[-1, 1]
)
fig.show()
# %%
import altair as alt
import pandas as pd
import numpy as np

x = np.random.lognormal(mean=np.log10(2.), sigma=1, size=100000)
y = np.random.lognormal(mean=np.log10(2.), sigma=1, size=100000)
df = pd.DataFrame({'x': x, 'y': y})

hist, xedges, yedges = np.histogram2d(df['x'], df['y'],
    bins=[np.logspace(np.log10(min(df['x'])), np.log10(max(df['x'])), 50),
        np.logspace(np.log10(min(df['y'])), np.log10(max(df['y'])), 50)])

df_binned = pd.DataFrame({'x': xedges[:-1], 'y': yedges[:-1], 'count': hist.ravel()})

chart = alt.Chart(df_binned).mark_rect().encode(
    x=alt.X('x:Q', scale=alt.Scale(type='log')),
    y=alt.Y('y:Q', scale=alt.Scale(type='log')),
    color=alt.Color('count:Q', scale=alt.Scale(scheme='greenblue'))
).properties(
    width=600,
    height=600,
    title='Heatmap of Log-Normal Distribution'
)

chart.show()


# %%
import pandas as pd
import numpy as np
import altair as alt

x = np.random.lognormal(mean=np.log10(2.), sigma=1, size=100000)
y = np.random.lognormal(mean=np.log10(2.), sigma=1, size=100000)
df = pd.DataFrame({'x': x, 'y': y})

hist, xedges, yedges = np.histogram2d(
    df['x'],
    df['y'],
    bins=[
        np.logspace(np.log10(min(df['x'])), np.log10(max(df['x'])), 200),
        np.logspace(np.log10(min(df['y'])), np.log10(max(df['y'])), 200)
    ]
)

# Create meshgrid of bin edges for x and y
x_centers = (xedges[:-1] + xedges[1:]) / 2
y_centers = (yedges[:-1] + yedges[1:]) / 2
X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')

df_binned = pd.DataFrame({
    'x': X.ravel(),
    'y': Y.ravel(),
    'count': hist.ravel()
})

# Now you can plot with Altair
chart = alt.Chart(df_binned).mark_rect(clip=True).encode(
    x=alt.X('x:Q', scale=alt.Scale(type='log')),
    y=alt.Y('y:Q', scale=alt.Scale(type='log')),
    color=alt.Color('count:Q', scale=alt.Scale(scheme='greenblue'))
).properties(
    width=600,
    height=600,
    title='Heatmap of Log-Normal Distribution'
)

chart.show()
# %%
import matplotlib.pyplot as plt
import numpy as np

mu, sigma = np.log10(3.), 1. # mean and standard deviation
s = np.random.lognormal(mu, sigma, 1000)

count, bins, ignored = plt.hist(s, 100, density=True, align='mid')
x = np.linspace(min(bins), max(bins), 10000)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
/ (x * sigma * np.sqrt(2 * np.pi)))
plt.plot(x, pdf, linewidth=2, color='r')
plt.axis('tight')
plt.xlim([0, 10])
plt.show()
# %%
# * This way can also be uused. to create a rasterize of points then get the 
# * rasterized image and plot it with hv.image and set the bounds and use log scale. 
# * its just no dynamic any more. 
# * To keep it interactive I can use rasterize with the log of the data and then
# * set the costume ticks and labels for the x and y axes. 
import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import (
    datashade,
    dynspread,
    rasterize,
    shade,
    spread,
)

from bokeh.models import FixedTicker

hv.extension('bokeh') #type: ignore
rng = np.random.default_rng()
x = rng.lognormal(mean=np.log(100.), sigma=np.log(5), size=100000)
y = rng.lognormal(mean=np.log(1.), sigma=np.log(2.), size=100000)
df = pd.DataFrame({'x': x, 'y': y})
logdf = np.log(df + 1e-3)  # Add a small value to avoid log(0)
points = hv.Points(logdf, kdims=['x', 'y'])

# Define ticks at log10 positions, but label as powers of 10
x_ticks = [(np.log(i), f"{i}") for i in [0.1, 1, 10, 100, 1000, 10000]]  # 1, 10, 100, 1000
y_ticks = [(np.log(i), f"{i}") for i in [0.1, 1, 10, 100, 100]]  # 0.1, 1, 10, 100
majorx_ticks = [np.log(i) for i in [0.1, 1, 10, 100, 1000, 10000]]  # 1, 10, 100, 1000
minorx_ticks = [np.log(i) for i in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]  # 0.1, 1, 10, 100
majorx_labels = [f"{i}" for i in [0.1, 1, 10, 100, 1000, 10000]]  # 1, 10, 100, 1000

datashaded_points = rasterize(points)
datashaded_points.opts(
    width=600,
    height=600,
    title='Log-Normal Distribution Heatmap',
    xlabel='X',
    ylabel='Y',
    # xaxis=None,
    # yaxis=None,
    # logx=True,
    # logy=True,
    # xlim=(-1, 1),
    # ylim=(-1, 1)
    # bounds=(0.1, 0.1, 10, 10),  # Set bounds for the image
    xticks=x_ticks,
    yticks=y_ticks,
    backend_opts={"plot.output_backend": "svg",}
)

# im_points = hv.Image(datashaded_points.data, 
#                     bounds=(-1, -1, 1, 1)).opts(
#     width=600,
#     height=600,
#     # title="Log-Normal Distribution Heatmap",
#     # xlabel="X",
#     # ylabel="Y",
#     # logx=True,
#     # logy=True,
#     # xlim=(-1, 1),
#     # ylim=(-1, 1),
# )
img = datashaded_points[()]
# datashaded_points
bounds = (
    np.exp(min(logdf['x'])),
    np.exp(min(logdf['y'])),
    np.exp(max(logdf['x'])),
    np.exp(max(logdf['y'])),
)

hv.Image(img.data['x_y Count'].values, bounds=bounds).opts(logx=True, logy=True)
# %%
rng = np.random.default_rng()
x = rng.lognormal(mean=np.log(50), sigma=np.log(2.), size=100000)
y = rng.lognormal(mean=np.log(3), sigma=np.log(2.), size=100000)
df = pd.DataFrame({'x': x, 'y': y})
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300  # Set default DPI

plt.figure(figsize=(6, 6))
plt.hist2d(df['x'], df['y'],
           bins=[np.logspace(np.log10(df['x'].min()), np.log10(df['x'].max()), 100),
                 np.logspace(np.log10(df['y'].min()), np.log10(df['y'].max()), 100)])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0.1, 100])
plt.ylim([0.1, 100])
plt.title('2D Histogram of Log-Normal Distribution')

plt.tight_layout()
plt.show()
# %%
data = [(i, chr(97+j),  i*j) for i in range(5) for j in range(5) if i!=j]
# %%
# %%
