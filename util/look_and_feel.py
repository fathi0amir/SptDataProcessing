"""
A place to put all the plot styling to visualize settings.
"""

# Import necessary libraries
import plotly.express as px
import numpy as np
import pandas as pd
import altair as alt
import vegafusion  # noqa: F401
import holoviews as hv
hv.extension('bokeh')  # type: ignore
# from plotly_resampler import FigureResampler, FigureWidgetResampler
# from plotly_resampler import register_plotly_resampler, unregister_plotly_resampler

# Enable VegaFusion for server-side transforms
alt.data_transformers.enable("vegafusion")
alt.data_transformers.disable_max_rows()

# Import custom module
import util.constants as const

def plotly_style_tracks(fig, px_size=65, img_size=900):
    fig.update_layout(
        width=600, 
        height=600,
        showlegend=False, 
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgb(52, 52, 52)',
    )
    fig.update_xaxes(
        range=[0, px_size * img_size], 
        title=None, 
        showticklabels=False, 
        showgrid=False, 
        zeroline=False
        )
    fig.update_yaxes(
        range=[0, px_size * img_size],
        title=None, 
        showticklabels=False, 
        showgrid=False, 
        zeroline=False
        )
    # Original figure is 800 pixels with 65nm per pixel, so 52000 is the max value
    # fig.update_layout(yaxis_scaleanchor="x", yaxis_scaleratio=1)
    # Show the plot
    
    return fig

def set_plotly_config(fig, width=800, height=600):
    """
    Set the configuration for the Plotly figure.
    Wrapper for fig.show(config=config)
    """
    config = {
        'toImageButtonOptions': {
            'scale': 1,
            'format': 'svg',
            'filename': 'figure',
            'width': width,
            'height': height
        }
    }
    return fig.show(config=config)

def plotly_style_single_track(fig):
    fig.update_layout(
        width=600, 
        height=600,
        showlegend=False, 
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgb(52, 52, 52)',
        yaxis_scaleanchor="x",  # lock aspect ratio
        yaxis_scaleratio=1      # 1:1 aspect ratio
    )
    fig.update_xaxes(
        title=None, 
        showticklabels=False, 
        showgrid=False, 
        zeroline=False
    )
    fig.update_yaxes(
        title=None, 
        showticklabels=False, 
        showgrid=False, 
        zeroline=False
    )
    return fig

def plotly_plot_diff_coef_hist(df, column='D_Fixed_Alpha'):
    """
    Plot the diffusion coefficient Histogram.
    """
    grouped_df = df.groupby('UID')[column].first().reset_index()
    # grouped_df = grouped_df[grouped_df['D_Fixed_Alpha'] > 0.1]
    fig = px.histogram(x=grouped_df[column], nbins=150)
    fig.update_layout(
        xaxis_title='Diffusion Coefficient (D)',
        yaxis_title='Count',
        title='Diffusion Coefficient Histogram',
        width=800,
        height=600,
        bargap=0.01,
        # xaxis_range=[0.1, 1],
        # yaxis_range=[0, 2],
        # paper_bgcolor='rgba(255, 255, 255, 0.90)',
        # plot_bgcolor='rgba(60, 60, 60, 0.44)'
        template='plotly_white',
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

    return set_plotly_config(fig)

def plotly_plot_alpha_hist(df, column='Alpha'):
    """
    Plot the diffusion coefficient Histogram.
    """
    grouped_df = df.groupby('UID')[column].first().reset_index()
    fig = px.histogram(x=grouped_df[column], nbins=50)
    fig.update_layout(
        xaxis_title='Alpha',
        yaxis_title='Count',
        title='Alpha Histogram',
        width=800,
        height=600,
        bargap=0.01,
        # xaxis_range=[0.1, 1],
        # yaxis_range=[0, 2],
        # paper_bgcolor='rgba(255, 255, 255, 0.90)',
        # plot_bgcolor='rgba(60, 60, 60, 0.44)'
        template='plotly_white',
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

    return set_plotly_config(fig)

def plotly_plot_norm_loglog_msd(df):
    """
    Plot the normalized log-log MSD.
    """
    # fig = FigureWidgetResampler(px.line())
    fig = px.line()
    for uid, group in df.groupby('UID'):
        fig.add_scatter(
            x=group['Lag_T'].iloc[0:6], 
            y=group['MSD'].iloc[0:6] / group['MSD'].iloc[0],
            mode='lines', 
            name=uid,
            line=dict(color='blue', width=1),
            opacity=const.OPACITY_PARAM/len(df['UID'].unique()) # 16 is just a number so for my testing set with 321 UIDs, this gives ∼0.05 opacity
            )

    fig.update_layout(
        template='plotly_white',
        xaxis_title='Lag Time (s)',
        yaxis_title='Mean Squared Displacement (MSD)',
        title='MSD vs Lag_T for all FileIDs and TrackIDs',
        width=800,
        height=600,
        # xaxis_range=[0.01, 0.2],
        # yaxis_range=[0.01, None],
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

    return fig

def plotly_plot_diff_coef_logloghist(df, column='D_Fixed_Alpha'):

    grouped_df = df.groupby('UID')[column].first().reset_index()
    hist, bin = np.histogram(grouped_df[column], bins=150)
    bin_centers = 0.5 * (bin[:-1] + bin[1:])

    fig = px.bar(x=bin_centers, y=hist)
    fig.update_traces(width=0.1)  # Adjust the bar width to fix the bar size
    fig.update_layout(
        xaxis_title='Diffusion Coefficient (D)',
        yaxis_title='Count',
        title='Diffusion Coefficient Histogram (Log-Log Scale)',
        width=800,
        height=600,
        xaxis_range=[np.log10(0.1), np.log10(10)],
        yaxis_range=[np.log10(1), np.log10(20)],
        xaxis_type='log',
        yaxis_type='log',
        template='plotly_white',
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
    return set_plotly_config(fig)

def plotly_plot_diff_coef_loglogarea(df, column='D_Fixed_Alpha'):
    """
    Plot the diffusion coefficient Histogram with log-log area.
    """
    grouped_df = df.groupby('UID')[column].first().reset_index()
    hist, bin = np.histogram(
        grouped_df[column], 
        bins=
        np.logspace(np.log(min(grouped_df[column])), np.log(max(grouped_df[column])), 30)
    )
    bin_centers = 0.5 * (bin[:-1] + bin[1:])
    fig = px.area(x=bin_centers, y=hist)
    fig.update_traces(fill='tozeroy')  # Fill the area under the curve
    fig.update_layout(
        xaxis_title='Diffusion Coefficient (µm²/s)',
        yaxis_title='Count',
        title='Diffusion Coefficient Histogram (Log-Log Scale)',
        width=800,
        height=600,
        xaxis_range=[np.log10(0.001), np.log10(10)],
        # yaxis_range=[np.log10(1), np.log10(50)],
        xaxis_type='log',
        # yaxis_type='log',
        template='plotly_white',
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
    return set_plotly_config(fig)

def plotly_plot_norm_msd_grouped(df, alphas):
    """
    Plot the normalized MSD grouped by UID.
    """

    fig = px.line()

    for uid, group in df.groupby('UID'):
        color = 'green'
        if group['Alpha_Flag_Fit'].iloc[0] == 'ignore':
            color = 'black'
        elif group['Alpha_Flag_Fit'].iloc[0] == 'sub':
            color = 'blue'
        elif group['Alpha_Flag_Fit'].iloc[0] == 'sup':
            color = 'red'

        fig.add_scatter(
            x=group['Lag_T'], 
            y=group['MSD'].iloc[0:6] / group['MSD'].iloc[0],
            mode='lines', 
            name=uid,
            line=dict(color=color, width=5),
            opacity=const.OPACITY_PARAM/len(df['UID'].unique()) # 16 (const.OPACITY_PARAM) is just a number so for my testing set with 321 UIDs, this gives ∼0.05 opacity
            )

        # Add three plots for different alpha values from alphas DataFrame
        for alpha_flag, alpha_value in alphas.items():
            color = 'green' if alpha_flag == 'normal' else 'blue' if alpha_flag == 'sub' else 'red' if alpha_flag == 'sup' else 'grey'
            msd_trend = np.array(range(1, 7)) ** alpha_value  # Assuming the first 6 points are used for MSD
            fig.add_scatter(
                x=group['Lag_T'].iloc[0:6],
                y=msd_trend,
                mode='lines',
                name=f'{alpha_flag} (Alpha={alpha_value:.2f})',
                line=dict(color=color, width=2),
                opacity=1
            )

    fig.update_layout(
        xaxis_title='Lag Time (s)',
        yaxis_title='Mean Squared Displacement (MSD)',
        title='MSD vs Lag_T for all FileIDs and TrackIDs',
        width=800,
        height=600,
        # xaxis_range=[0.01, 0.2],
        # yaxis_range=[0.01, 2.5],
        xaxis_type='log',
        yaxis_type='log', 
        showlegend=False,
        template='plotly_white',
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

    return fig

def plotly_plot_diff_coef_vs_alpha(df):
    """
    Plot the diffusion coefficient vs alpha.
    """
    grouped_df = df.groupby('UID')[['D_Fixed_Alpha', 'Alpha', 'Alpha_Flag_Fit']].first().reset_index()
    fig = px.scatter(
        grouped_df, 
        x='Alpha', 
        y='D_Fixed_Alpha', 
        color='Alpha_Flag_Fit',
        color_discrete_map={
            'ignore': 'black',
            'sub': 'blue',
            'sup': 'red',
            'normal': 'green'
        }
    )
    fig.update_layout(
        xaxis_title='Alpha',
        yaxis_title='Diffusion Coefficient (µm²/s)',
        title='Diffusion Coefficient vs Alpha',
        width=800,
        height=600,
        # xaxis_range=[-0.5, 6],
        # yaxis_range=[-0.5, 6],
        # paper_bgcolor='rgb(255, 255, 255)',
        # plot_bgcolor='rgb(220, 220, 220)',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            x=0,
            y=1,
            xanchor='left',
            yanchor='top', 
            title='Alpha Flag',
        ),
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

    return set_plotly_config(fig)

def vega_plot_msd_loglog_fast(df, bin_size=0.03):
    """
    Plot the normalized log-log MSD using Vega.
    """
    
    # Create the chart
    chart = alt.Chart(df).transform_calculate(
        log_x='log(datum.Lag_T)/log(10)',
        log_y='log(datum.MSD_norm)/log(10)'
    ).transform_bin(
        field='log_x',
        as_=['bin_log_x', 'bin_log_x_end'],
        bin=alt.Bin(maxbins=200, step=bin_size, base=10)
    ).transform_bin(
        field='log_y',
        as_=['bin_log_y', 'bin_log_y_end'],
        bin=alt.Bin(maxbins=200, step=bin_size, base=10)
    ).transform_calculate(
        x1='pow(10, datum.bin_log_x)',
        x2='pow(10, datum.bin_log_x_end)',
        y1='pow(10, datum.bin_log_y)',
        y2='pow(10, datum.bin_log_y_end)'
    ).mark_rect(clip=True).encode(
        x=alt.X('x1:Q', 
                scale=alt.Scale(type='log', base=10),
                axis=alt.Axis(tickCount=5)),
        x2='x2:Q',
        y=alt.Y('y1:Q', 
                scale=alt.Scale(type='log', base=10),
                axis=alt.Axis(tickCount=5)),
        y2='y2:Q',
        color=alt.Color('count():Q', scale=alt.Scale(scheme='greenblue', domain=[0, 150])),
        tooltip=[
            alt.Tooltip('x1:Q', title='X Bin Start'),
            alt.Tooltip('x2:Q', title='X Bin End'),
            alt.Tooltip('y1:Q', title='Y Bin Start'),
            alt.Tooltip('y2:Q', title='Y Bin End'),
            alt.Tooltip('count():Q', title='Count')
        ]
    ).properties(
        description='Log-scaled Histogram.',
        width=600,
        height=600
    ).interactive()

    return chart.show()

def holoviz_plot_msd_loglog_fast(df, bin_num=200):
    """
    Plot the normalized log-log MSD using HoloViz.
    """
    hist, xedges, yedges = np.histogram2d(
        df.Lag_T,
        df.MSD_norm,
        bins=[
            np.logspace(np.log(min(df.Lag_T)), np.log(max(df.Lag_T)), 200),
            np.logspace(np.log(min(df.MSD_norm)), np.log(max(df.MSD_norm)), 200),
        ],
    )
    hist[hist == 0] = np.nan  # Replace zeros with NaN for better visualization
    heatmap = hv.HeatMap((xedges[:-1], yedges[:-1], hist.T)).opts(
        width=600,
        height=600,
        title='Log-Normal Distribution Heatmap',
        xlabel='Time Lag (s)',
        ylabel='MSD (µm²)',
        logx=True,
        logy=True,
        xlim=(0.01, 1000),
        ylim=(0.01, 1000),
        clim=(0, 30),  # Set color limits for the heatmap
        colorbar=True,  # Show color bar
        cmap='viridis'
    )

    return heatmap