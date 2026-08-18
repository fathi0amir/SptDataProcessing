"""
A place to put all the plot styling to visualize settings.
"""
from copy import copy
# Import necessary libraries
import plotly.express as px
import numpy as np
import pandas as pd
import altair as alt
import vegafusion  # noqa: F401
import holoviews as hv
from bokeh.themes import Theme
from bokeh.models import LinearAxis, LogAxis

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

def plotly_plot_diff_coef_hist(df, column='D_Fixed_Alpha', nbins=150):
    """
    Plot the diffusion coefficient Histogram.
    """
    grouped_df = df.groupby('UID')[column].first().reset_index()
    # grouped_df = grouped_df[grouped_df['D_Fixed_Alpha'] > 0.1]
    fig = px.histogram(x=grouped_df[column], nbins=nbins)
    fig.update_layout(
        xaxis_title='Diffusion Coefficient (µm²/s)',
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
    hist, bin = np.histogram(grouped_df[column], bins=1000)
    bin_centers = 0.5 * (bin[:-1] + bin[1:])

    fig = px.bar(x=bin_centers, y=hist)
    fig.update_traces(width=0.1)  # Adjust the bar width to fix the bar size
    fig.update_layout(
        xaxis_title='Diffusion Coefficient (D)',
        yaxis_title='Count',
        title='Diffusion Coefficient Histogram (Log-Log Scale)',
        width=800,
        height=600,
        xaxis_range=[np.log10(0.001), np.log10(0.1)],
        yaxis_range=[np.log10(1), np.log10(1000)],
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
        np.logspace(np.log(min(grouped_df[column])), np.log(max(grouped_df[column])), 100)
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
        xaxis_range=[np.log10(0.0001), np.log10(0.01)],
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


# Defualt DPI for web standard is 96 (can be stimates as 100 pixels per 25 mm)
HV_BOKEH_BASIC = hv.opts(
    width=800,
    height=600,
    fontsize=12,
    fontscale=2.0,
    # logx=False,
    # logy=False,
    backend_opts={
        "plot.output_backend": "svg",
        # "plot.title.text_font": "Noto Sans",
        # "plot.background_fill_color": '#2F2F2F',
        # "plot.border_fill_color": '#2F2F2F',
        # "plot.outline_line_color": '#444444',
        # "axis.axis_line_color": None,
        # "plot.xaxis.axis_label_text_font": "Noto Sans",
        # "plot.yaxis.axis_label_text_font": "Noto Sans",
        # "plot.xaxis.major_label_text_font": "Noto Sans",
        # "plot.yaxis.major_label_text_font": "Noto Sans",
        # "plot.legend.label_text_font": "Noto Sans",
        # "plot.legend.title_text_font": "Noto Sans",
    },
)

HV_BOKEH_CURVE = hv.opts.Curve(
    line_width=2,
)

def bokeh_remove_bottomleft_labels(plot, element):
    p = plot.handles['plot']
    p.xaxis[0].major_label_text_font_size = "0pt"  # Hide x-axis labels
    p.yaxis[0].major_label_text_font_size = "0pt"  # Hide y-axis labels
    p.xaxis[0].major_label_text_alpha = 0  # Hide x-axis labels
    p.yaxis[0].major_label_text_alpha = 0  # Hide y-axis labels

def bokeh_add_topright_linear_axes(plot, element):
    p = plot.handles['plot']

    # Clone ticker and formatter from existing axes
    top_axis = LinearAxis(
        ticker=copy(p.xaxis[0].ticker),
        formatter=copy(p.xaxis[0].formatter),
        major_label_text_font_size="0pt",  # Hide labels on the top axis
        # axis_label=p.xaxis[0].axis_label,
        major_label_text_alpha = 0
    )
    right_axis = LinearAxis(
        ticker=copy(p.yaxis[0].ticker),
        formatter=copy(p.yaxis[0].formatter),
        major_label_text_font_size="0pt",  # Hide labels on the right axis
        # axis_label=p.yaxis[0].axis_label,
        major_label_text_alpha = 0
    )

    p.add_layout(top_axis, 'above')
    p.add_layout(right_axis, 'right')

def bokeh_add_topright_log_axes(plot, element):
    p = plot.handles['plot']

    # Clone ticker and formatter from existing axes
    top_axis = LogAxis(
        ticker=copy(p.xaxis[0].ticker),
        formatter=copy(p.xaxis[0].formatter),
        major_label_text_font_size="0.01pt",  # Hide labels on the top axis
        major_label_text_alpha=0,  # Hide labels on the top axis
        # axis_label=p.xaxis[0].axis_label,
    )
    right_axis = LogAxis(
        ticker=copy(p.yaxis[0].ticker),
        formatter=copy(p.yaxis[0].formatter),
        major_label_text_font_size="0.01pt",  # Hide labels on the right axis
        major_label_text_alpha=0,  # Hide labels on the right axis
        # axis_label=p.yaxis[0].axis_label,
    )

    p.add_layout(top_axis, 'above')
    p.add_layout(right_axis, 'right')



def bokeh_add_topright_logxliny_axes(plot, element):
    fig = plot.state

    # Right axis — mirrors the (linear) y-range
    fig.extra_y_ranges = {'y2': fig.y_range}
    right_axis = LinearAxis(
        y_range_name='y2', 
        ticker=fig.yaxis[0].ticker, 
        major_label_text_font_size='1pt',
        major_label_text_alpha=0)
    fig.add_layout(right_axis, 'right')

    # Top axis — mirrors the (log) x-range
    fig.extra_x_ranges = {'x2': fig.x_range}
    top_axis = LogAxis(
        x_range_name='x2', 
        ticker=fig.xaxis[0].ticker, 
        major_label_text_font_size='1pt', 
        major_label_text_alpha=0)
    fig.add_layout(top_axis, 'above')



def move_axes_topleft(plot, element):
    """Hook to add mirrored top and right axes."""
    p = plot.handles['plot']  # Get the underlying Bokeh figure
    
    # Add right axis (mirrors left/y-axis)
    p.add_layout(p.yaxis[0], 'right')   # or LinearAxis(), if you want a separate one
    
    # Add top axis (mirrors bottom/x-axis)
    p.add_layout(p.xaxis[0], 'above')


def set_legend_top_horizontal(plot, element):
    p = plot.handles['plot']
    if p.legend:
        p.legend[0].orientation = 'horizontal'
        p.legend[0].location = 'top_center'
        p.legend[0].title=''



def single_mol_single_step_photobleaching_dashboard(df, bleach_df, fit_summary, intensity_col='mass'):
    """
    Generates a 3x2 HoloViews dashboard for single-step photobleaching traces and fits.
    """
    if bleach_df.empty:
        return None

    hv.renderer("bokeh").theme = Theme(filename="util/bokeh-theme-light.yaml")
    all_tracks = df.groupby('UID')

    # Select representative tracks
    sorted_by_snr = bleach_df.sort_values('snr', ascending=False)
    rep_indices = [0, len(bleach_df) // 4, len(bleach_df) // 2, -1]
    rep_tracks = sorted_by_snr.iloc[rep_indices]
    
    trace_plots = []
    for idx, (_, row) in enumerate(rep_tracks.iterrows()):
        track_data = all_tracks.get_group(row['UID']).sort_values('Frame')
        t_time = (track_data['Frame'].values - track_data['Frame'].values[0]) * const.DT
        t_intensities = track_data[intensity_col].values

        raw_curve = hv.Curve((t_time, t_intensities), kdims=['Time (s)'], vdims=['Intensity (a.u.)']).opts(line_color='#5A96DC', line_alpha=0.4)
        raw_scatter = hv.Scatter((t_time, t_intensities), kdims=['Time (s)'], vdims=['Intensity (a.u.)']).opts(color='#5A96DC', size=4.5, alpha=0.85)

        step_time = t_time[row['step_idx']]
        step_t = [t_time[0], step_time, step_time, t_time[-1]]
        step_y = [row['mean_before'], row['mean_before'], row['mean_after'], row['mean_after']]
        fit_curve = hv.Curve((step_t, step_y), kdims=['Time (s)'], vdims=['Intensity (a.u.)']).opts(line_color='#E74C3C', line_width=2.5)

        label_text = f"UID: {row['UID']}\nΔI = {row['step_size']:.0f}\nSNR = {row['snr']:.1f}"
        text_ann = hv.Text(t_time[-1] * 0.5, max(t_intensities) * 0.85, label_text, fontsize=10)

        p = (raw_curve * raw_scatter * fit_curve * text_ann).opts(
            title=f"Representative Track {idx+1}", width=380, height=280
        ).opts(HV_BOKEH_BASIC).opts(hooks=[bokeh_add_topright_linear_axes])
        trace_plots.append(p)

    # Step Size Histogram Plot
    counts, bin_edges, popt_gauss = fit_summary['gauss']
    step_hist = hv.Histogram((counts, bin_edges), kdims=['Intensity Drop ΔI (a.u.)'], vdims=['Frequency'], label='Observed Steps').opts(color='#2ECC71', alpha=0.55)
    if popt_gauss is not None:
        fit_x = np.linspace(min(bleach_df['step_size']), max(bleach_df['step_size']), 150)
        fit_y = popt_gauss[0] * np.exp(-(fit_x - popt_gauss[1])**2 / (2 * popt_gauss[2]**2))
        step_plot = step_hist * hv.Curve((fit_x, fit_y), label=f'Gaussian (μ={popt_gauss[1]:.1f}, σ={popt_gauss[2]:.1f})').opts(line_color='#2ECC71', line_width=3.0)
    else:
        step_plot = step_hist
    step_plot = step_plot.opts(title="Bleaching Step Size", width=380, height=280, legend_position='top_right').opts(HV_BOKEH_BASIC).opts(hooks=[bokeh_add_topright_linear_axes, set_legend_top_horizontal])

    # Survival Time Histogram Plot
    counts_t, bin_edges_t, popt_exp = fit_summary['exp']
    time_hist = hv.Histogram((counts_t, bin_edges_t), kdims=['Survival Time Before Bleach (s)'], vdims=['Frequency'], label='Survival Time').opts(color='#9B59B6', alpha=0.55)
    if popt_exp is not None:
        fit_t = np.linspace(0, max(bleach_df['bleach_time']), 150)
        fit_n = popt_exp[0] * np.exp(-fit_t / popt_exp[1])
        time_plot = time_hist * hv.Curve((fit_t, fit_n), label=f'Exp. Decay (τ={popt_exp[1]:.2f} s)').opts(line_color='#9B59B6', line_width=3.0)
    else:
        time_plot = time_hist
    time_plot = time_plot.opts(title="Photobleaching Lifetime Decay", width=380, height=280, legend_position='top_right').opts(HV_BOKEH_BASIC).opts(hooks=[bokeh_add_topright_linear_axes, set_legend_top_horizontal])

    return hv.Layout([*trace_plots, step_plot, time_plot]).cols(2).opts(
        title="Single-Molecule Proof: Single-Step Photobleaching Dashboard", shared_axes=False, axiswise=True, framewise=True
    )

def single_mol_multi_step_photobleaching_dashboard(df, results_df, intensity_col='mass'):
    """
    Generates a 3x2 HoloViews dashboard for multi-step photobleaching traces and stoichiometry.
    """
    if results_df.empty:
        return None

    hv.extension("bokeh")
    hv.renderer("bokeh").theme = Theme(filename="util/bokeh-theme-light.yaml")
    all_tracks = df.groupby('UID')

    sorted_df = results_df.sort_values(by=['num_steps', 'final_ratio'], ascending=[False, True])
    rep_uids = []
    
    for steps in [3, 2, 2, 1]:
        sub = sorted_df[sorted_df['num_steps'] == steps]
        if not sub.empty:
            candidate = sub.iloc[0]['UID']
            if candidate not in rep_uids:
                rep_uids.append(candidate)
                
    for _, row in sorted_df.iterrows():
        if len(rep_uids) >= 4:
            break
        if row['UID'] not in rep_uids:
            rep_uids.append(row['UID'])
            
    trace_plots = []
    
    for idx, uid in enumerate(rep_uids):
        fit_info = results_df[results_df['UID'] == uid].iloc[0]
        track_data = all_tracks.get_group(uid).sort_values('Frame')
        t_frames = track_data['Frame'].values
        t_time = (t_frames - t_frames[0]) * const.DT
        t_intensities = track_data[intensity_col].values
        
        raw_curve = hv.Curve((t_time, t_intensities), kdims=['Time (s)'], vdims=['Intensity (a.u.)']).opts(color='#5A96DC', line_width=1.0, alpha=0.4)
        raw_scatter = hv.Scatter((t_time, t_intensities), kdims=['Time (s)'], vdims=['Intensity (a.u.)']).opts(color='#5A96DC', size=4.5, alpha=0.85)
        
        locs = fit_info['step_locations_idx']
        edges_idx = [0] + locs + [len(track_data)]
        step_t, step_y = [], []
        for i in range(len(edges_idx) - 1):
            s_idx, e_idx = edges_idx[i], edges_idx[i+1]
            step_t.append(t_time[s_idx])
            step_y.append(fit_info['plateau_values'][i])
            step_t.append(t_time[min(e_idx - 1, len(t_time) - 1)])
            step_y.append(fit_info['plateau_values'][i])
            
        fit_curve = hv.Curve((step_t, step_y), kdims=['Time (s)'], vdims=['Intensity (a.u.)']).opts(color='#E74C3C', line_width=2.5)
        
        text_x = t_time[-1] * 0.5
        text_y = max(t_intensities) * 0.85
        step_sizes_formatted = ", ".join([f"{int(s)}" for s in fit_info['step_sizes']])
        label_text = f"UID: {uid} \nSteps: {fit_info['num_steps']}\nΔI: [{step_sizes_formatted}]"
        
        text_annotation = hv.Text(text_x, text_y, label_text, fontsize=10).opts(text_color='black')
        
        title_classes = ["Monomer", "Dimer", "Trimer/Aggregate"]
        class_label = title_classes[min(fit_info['num_steps'] - 1, 2)]
        
        combined_track = (raw_curve * raw_scatter * fit_curve * text_annotation).opts(
            title=f"Representative {class_label} Trace", width=380, height=280
        ).opts(HV_BOKEH_BASIC).opts(hooks=[bokeh_add_topright_linear_axes])
        
        trace_plots.append(combined_track)
        
    counts = results_df['num_steps'].value_counts().sort_index()
    proportions = (counts / counts.sum() * 100).round(1)
    
    proportion_df = pd.DataFrame({
        'Steps': [f"{s} Step" + ("s" if s > 1 else "") for s in counts.index],
        'Percent': proportions.values
    })
    
    proportion_bars = hv.Bars(proportion_df, kdims='Steps', vdims='Percent').opts(
        title='Assembly Stoichiometry Distribution', xlabel='Bleaching Profile', ylabel='Proportion of Tracks (%)', color='#4E79A7', width=380, height=280
    ).opts(HV_BOKEH_BASIC).opts(hooks=[bokeh_add_topright_linear_axes])
    
    all_individual_steps = []
    for sizes in results_df['step_sizes']:
        all_individual_steps.extend(sizes)
        
    step_sizes_arr = np.array(all_individual_steps)
    counts_sz, bin_edges_sz = np.histogram(step_sizes_arr, bins='auto')
    
    size_hist = hv.Histogram((counts_sz, bin_edges_sz), kdims=['Bleaching Step Size (a.u.)'], vdims=['Frequency']).opts(
        title='Individual Step Size Dev (Bleaching Unit)', color='#2ECC71', alpha=0.55, width=380, height=280
    ).opts(HV_BOKEH_BASIC).opts(hooks=[bokeh_add_topright_linear_axes])
    
    return hv.Layout(
        [*trace_plots, proportion_bars, size_hist]
    ).cols(2).opts(
        title="Oligomerization Stoichiometry & Multi-Step Bleaching Dashboard",
        shared_axes=False, axiswise=True, framewise=True
    )


def single_mol_multi_step_photobleaching_dashboard_tabbed(df, results_df, intensity_col='mass'):
    """
    Generates a HoloViews Tabs (hv.Tabs) dashboard for multi-step photobleaching traces and stoichiometry.
    """
    if results_df.empty:
        return None

    hv.extension("bokeh")
    hv.renderer("bokeh").theme = Theme(filename="util/bokeh-theme-light.yaml")
    all_tracks = df.groupby('UID')

    sorted_df = results_df.sort_values(by=['num_steps', 'final_ratio'], ascending=[False, True])
    rep_uids = []
    
    for steps in [3, 2, 2, 1]:
        sub = sorted_df[sorted_df['num_steps'] == steps]
        if not sub.empty:
            candidate = sub.iloc[0]['UID']
            if candidate not in rep_uids:
                rep_uids.append(candidate)
                
    for _, row in sorted_df.iterrows():
        if len(rep_uids) >= 4:
            break
        if row['UID'] not in rep_uids:
            rep_uids.append(row['UID'])
            
    trace_plots = []
    
    for idx, uid in enumerate(rep_uids):
        fit_info = results_df[results_df['UID'] == uid].iloc[0]
        track_data = all_tracks.get_group(uid).sort_values('Frame')
        t_frames = track_data['Frame'].values
        t_time = (t_frames - t_frames[0]) * const.DT
        t_intensities = track_data[intensity_col].values

        raw_curve = hv.Curve(
            (t_time, t_intensities), kdims=["Time (s)"], vdims=["Intensity (a.u.)"]
        ).opts(line_color="#5A96DC", line_width=1.0, line_alpha=0.4, active_tools=[])
        raw_scatter = hv.Scatter(
            (t_time, t_intensities), kdims=["Time (s)"], vdims=["Intensity (a.u.)"]
        ).opts(color="#5A96DC", size=4.5, alpha=0.85, active_tools=[])
        
        locs = fit_info['step_locations_idx']
        edges_idx = [0] + locs + [len(track_data)]
        step_t, step_y = [], []
        for i in range(len(edges_idx) - 1):
            s_idx, e_idx = edges_idx[i], edges_idx[i+1]
            step_t.append(t_time[s_idx])
            step_y.append(fit_info['plateau_values'][i])
            step_t.append(t_time[min(e_idx - 1, len(t_time) - 1)])
            step_y.append(fit_info['plateau_values'][i])

        fit_curve = hv.Curve(
            (step_t, step_y), kdims=["Time (s)"], vdims=["Intensity (a.u.)"]
        ).opts(line_color="#E74C3C", line_width=2.5, active_tools=[])
        
        text_x = t_time[-1] * 0.5
        text_y = max(t_intensities) * 0.85
        step_sizes_formatted = ", ".join([f"{int(s)}" for s in fit_info['step_sizes']])
        label_text = f"UID: {uid} \nSteps: {fit_info['num_steps']}\nΔI: [{step_sizes_formatted}]"
        
        text_annotation = hv.Text(text_x, text_y, label_text, fontsize=10).opts(text_color='#333333')
        
        title_classes = ["Monomer", "Dimer", "Trimer/Aggregate"]
        class_label = title_classes[min(fit_info['num_steps'] - 1, 2)]
        
        combined_track = (raw_curve * raw_scatter * fit_curve * text_annotation).opts(
            title=f"{class_label} Trace {idx+1}", show_legend=False
        ).opts(HV_BOKEH_BASIC).opts(hooks=[bokeh_add_topright_linear_axes])
        
        trace_plots.append((f"Trace {idx+1}", combined_track))
        
    counts = results_df['num_steps'].value_counts().sort_index()
    proportions = (counts / counts.sum() * 100).round(1)
    
    proportion_df = pd.DataFrame({
        'Steps': [f"{s} Step" + ("s" if s > 1 else "") for s in counts.index],
        'Percent': proportions.values
    })

    proportion_bars = (
        hv.Bars(proportion_df, kdims="Steps", vdims="Percent")
        .opts(
            title="Bleach Type Distribution",
            xlabel="Bleaching Profile",
            ylabel="Proportion of Tracks (%)",
            color="#4E79A7",
            width=750,
            height=450,
            show_legend=False,
        )
        .opts(HV_BOKEH_BASIC)
        .opts(hooks=[bokeh_add_topright_linear_axes])
        .opts(active_tools=[])
    )
    
    all_individual_steps = []
    for sizes in results_df['step_sizes']:
        all_individual_steps.extend(sizes)
        
    step_sizes_arr = np.array(all_individual_steps)
    counts_sz, bin_edges_sz = np.histogram(step_sizes_arr, bins='auto')

    size_hist = (
        hv.Histogram(
            (counts_sz, bin_edges_sz),
            kdims=["Bleaching Step Size (a.u.)"],
            vdims=["Frequency"],
        )
        .opts(
            title="Step Size Distribution",
            color="#2ECC71",
            alpha=0.55,
            width=750,
            height=450,
            show_legend=False,
        )
        .opts(HV_BOKEH_BASIC)
        .opts(hooks=[bokeh_add_topright_linear_axes])
        .opts(active_tools=[])
    )

    tab_dict = {
        f"Trace {i+1}": plot for i, (_, plot) in enumerate(trace_plots)
    } | {
        "Stoichiometry": proportion_bars,
        "Step Size": size_hist,
    }

    tabs = hv.Layout(tab_dict).opts(
        tabs=True,
        shared_axes=False,
        axiswise=True,
        framewise=True
    )
    
    return tabs