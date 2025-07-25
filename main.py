# %% # MARK: Imports
from pathlib import Path
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
data_path = Path(r"F:\Amir\SLB_Flaoting_Dye_Training_Set")
df = dpp.load_csv_files(data_path)

df.info()
df.head(20)
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
#
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
# %% # * Calculate D when Alpha is fixed (exact value for that trajectory)
df = df.groupby('UID').apply(nlss.calc_d_fix_alpha).reset_index(drop=True)
#
#
#
# %% # * Create a new dataframe with two columns: Alpha Flag , mean Alpha for each flag (class)
alphas = df.groupby('Alpha_Flag_Fit')['Alpha'].mean()
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
# Plot a histogram of the 'D' column
reload_modules()
laf.plotly_plot_diff_coef_hist(df, column='D_Fixed_Alpha')
# laf.plotly_plot_diff_coef_logloghist(df)
# laf.plotly_plot_diff_coef_loglogarea(df)
#
#
#
# %% # * ====================================
# Plot a histogram of the 'Alpha' column
reload_modules()
laf.plotly_plot_alpha_hist(df)
# %% # * ====================================
# Plot MSD vs Lag_T for each FileID and TrackID
# MSDs are normalized by the first MSD value for each UID
# This is to compare the MSDs across all the trajectories
reload_modules()
laf.plotly_plot_norm_loglog_msd(df)
#
#
#
# %% # * ====================================
# Plot MSD vs Lag_T for each FileID and TrackID
# Color from the alpha flag
reload_modules()
laf.plotly_plot_norm_msd_grouped(df, alphas)
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
# MARK: Plot MSD JD
# Create a Holoviews Dataset for the MSD vs Lag_T plot
# Overlay all curves for each TrackID
msd_overlay = hv.NdOverlay({
    uid: hv.Curve(
        (group['Lag_T'].iloc[0:6],  # Use first 6 points for MSD
        group['MSD'].iloc[0:6] / group['MSD'].iloc[0]),  # Normalize by first value
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
        (group['Lag_T'], group['MSD']/ group['MSD'].iloc[0]),  # Normalize by first value
        label=str(uid)
    ).opts()
    for uid, group in df.groupby('UID')
}).opts() #type: ignore

# Display the plot
# spread(rasterize(paths), px=1) = rasterize(paths, line_width=2)
rasterize(msd_overlay, line_width=1).opts( #type: ignore
    alpha=1,
    title="MSD vs Lag_T for each UID",
    xlabel="Lag Time (s)",
    ylabel="Mean Squared Displacement (MSD)", logy=True, logx=True,
    xlim=(None, 0.2),
    ylim=(1, 12),
    width=800,
    height=600,
    toolbar='above',
    clim=(0, 40),
    backend_opts={"plot.output_backend": "svg"}) #type: ignore
# msd_overlay

#%%
#
# Create a heatmap of Lag_T vs MSD
fig = px.density_heatmap(x=df["Lag_T"].iloc[0:6], y=df["MSD"].iloc[0:6], title="Heatmap of Lag_T vs MSD", nbinsx=100, nbinsy=10)

fig.show()
#
#
#
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

df['MSD_norm'] = df.groupby('UID')['MSD'].transform(lambda x: x / x.iloc[0])

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
