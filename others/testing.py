# %%
import pandas as pd
import holoviews as hv
import numpy as np

# Enable Holoviews extension
hv.extension('bokeh')

# Generate some demo data
x = np.linspace(0, 30, 800)
y = np.sin(x)

# Create a Holoviews Curve plot
curve = hv.Curve((x, y), 'X-axis', 'Y-axis').opts(title="Demo Sine Wave Plot", width=600, height=400)

curve.opts(
    backend_opts={"plot.output_backend": "svg"}
)
# Display the plot
curve
# %%
# Playing with melt in pandas

df = pd.DataFrame({
    'x1': [1, 2, 3, None, None],
    'y1': [10, 20, 30, None, None],
    'x2': [4, 5, 6, 7, 8],
    'y2': [40, 50, 60, 70, 80]
})

# Melt and reshape
df_melted = pd.DataFrame({
    'x': pd.concat([df['x1'], df['x2']], ignore_index=True),
    'y': pd.concat([df['y1'], df['y2']], ignore_index=True),
    'group': ['1'] * len(df) + ['2'] * len(df)
})

print(df_melted)

# %%
# 
# This also handles log scaling for binning 
import numpy as np
import holoviews as hv

hv.extension('bokeh')  # Ensure Bokeh backend is used

def np10(n):
    """Find the next power of 10 greater than or equal to n"""
    if n <= 0:
        return 1
    
    # Calculate the exponent needed
    exponent = np.ceil(np.log10(n))
    
    # Return 10 raised to that exponent
    return 10 ** exponent

#prepare sample data
rng = np.random.default_rng()
x = rng.lognormal(mean=np.log(100.), sigma=np.log(100), size=1000000)
y = rng.lognormal(mean=np.log(0.1), sigma=np.log(1000), size=1000000)

binx_scale = np.std(x)/np.mean(x)*100
biny_scale = np.std(y)/np.mean(y)*100
binx_min = np10(min(x)/binx_scale)
binx_max = np10(max(x)*binx_scale)
biny_min = np10(min(y)/biny_scale)
biny_max = np10(max(y)*biny_scale)

hist, xedges, yedges = np.histogram2d(
    x,
    y,
    bins=[
        np.logspace(np.log(binx_min), np.log(binx_max), 300),
        np.logspace(np.log(biny_min), np.log(biny_max), 300),
    ],
)

heatmap = hv.HeatMap((xedges[:-1], yedges[:-1], hist.T)).opts(
    width=600,
    height=600,
    title='Log-Normal Distribution Heatmap',
    xlabel='X',
    ylabel='Y',
    logx=True,
    logy=True,
    # xlim=(1.1, 1000),
    # ylim=(0.0001, 1)
)

heatmap
# %%
import plotly.express as px

fig = px.imshow(
    hist.T,
    x=xedges[:-1],
    y=yedges[:-1],
    aspect="auto"
)
fig.update_xaxes(type='log', title_text='X-axis (log scale)', range=[np.log(min(x)), np.log(max(x))])
fig.update_yaxes(type='log', title_text='Y-axis (log scale)', range=[np.log(min(y)), np.log(max(y))])


fig.show()
# %% Testing vega-altair with log scaling
# binning transormations are done with vega as well
import altair as alt
import pandas as pd
import numpy as np
import vegafusion  # noqa: F401

# Enable VegaFusion for server-side transforms
alt.data_transformers.enable("vegafusion")
alt.data_transformers.disable_max_rows()

# Create the data

rng = np.random.default_rng()
x = rng.lognormal(mean=np.log(10.), sigma=np.log(2), size=1000000)
y = rng.lognormal(mean=np.log(1.), sigma=np.log(2), size=1000000)

data = pd.DataFrame({
    'x': x, 
    'y': y
})

# Create the chart
chart = alt.Chart(data).transform_calculate(
    log_x='log(datum.x)/log(10)',
    log_y='log(datum.y)/log(10)'
).transform_bin(
    field='log_x',
    as_=['bin_log_x', 'bin_log_x_end'],
    bin=alt.Bin(maxbins=500, step=0.01, base=10)
).transform_bin(
    field='log_y',
    as_=['bin_log_y', 'bin_log_y_end'],
    bin=alt.Bin(maxbins=500, step=0.01, base=10)
).transform_calculate(
    x1='pow(10, datum.bin_log_x)',
    x2='pow(10, datum.bin_log_x_end)',
    y1='pow(10, datum.bin_log_y)',
    y2='pow(10, datum.bin_log_y_end)'
).mark_rect().encode(
    x=alt.X('x1:Q', 
            scale=alt.Scale(type='log', base=10),
            axis=alt.Axis(tickCount=5)),
    x2='x2:Q',
    y=alt.Y('y1:Q', 
            scale=alt.Scale(type='log', base=10),
            axis=alt.Axis(tickCount=5)),
    y2='y2:Q',
    color=alt.Color('count():Q', scale=alt.Scale(scheme='greenblue')),
    tooltip=[
        alt.Tooltip('x1:Q', title='X Bin Start'),
        alt.Tooltip('x2:Q', title='X Bin End'),
        alt.Tooltip('y1:Q', title='Y Bin Start'),
        alt.Tooltip('y2:Q', title='Y Bin End'),
        alt.Tooltip('count():Q', title='Count')
    ]
).properties(
    description='Log-scaled Histogram.'
)

chart.show()



# %%
import altair as alt
import pandas as pd
import numpy as np

# Create the data
data = pd.DataFrame({
    'x': [0.01, 0.1, 1, 1, 1, 1, 10, 10, 100, 500, 800]
})

# Create the chart
chart = alt.Chart(data).transform_calculate(
    log_x='log(datum.x)/log(10)'
).transform_bin(
    field='log_x',
    as_=['bin_log_x', 'bin_log_x_end']
).transform_calculate(
    x1='pow(10, datum.bin_log_x)',
    x2='pow(10, datum.bin_log_x_end)'
).mark_bar().encode(
    x=alt.X('x1:Q', 
            scale=alt.Scale(type='log', base=10),
            axis=alt.Axis(tickCount=5)),
    x2='x2:Q',
    y=alt.Y(aggregate='count')
).properties(
    description='Log-scaled Histogram.'
)

chart.show()


# %%
# * This is how to do it use the heatmap with log scale and bin the data with numpy
import numpy as np
import holoviews as hv

hv.extension('bokeh')  # Ensure Bokeh backend is used

#prepare sample data
rng = np.random.default_rng()
x = rng.lognormal(mean=np.log(10.), sigma=np.log(2), size=1000000)
y = rng.lognormal(mean=np.log(1.), sigma=np.log(2.), size=1000000)

hist, xedges, yedges = np.histogram2d(
    x,
    y,
    bins=[
        np.logspace(np.log(min(x)), np.log(max(x)), 200),
        np.logspace(np.log(min(y)), np.log(max(y)), 200),
    ],
)

heatmap = hv.HeatMap((xedges[:-1], yedges[:-1], hist.T)).opts(
    width=600,
    height=600,
    title='Log-Normal Distribution Heatmap',
    xlabel='X',
    ylabel='Y',
    logx=True,
    logy=True,
    xlim=(0.01, 1000),
    ylim=(0.01, 1000)
)

heatmap