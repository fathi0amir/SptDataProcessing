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


