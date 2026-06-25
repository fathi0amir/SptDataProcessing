# SPT Toolkit to process SPT data

## Main packages I used

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations. Limit to 2.1 only (for datashader) testing
- **plotly**: For interactive visualizations.
- **plotly-resampler**: For resampling large datasets in Plotly visualizations.
- **holoviews**: For high-level data visualization.
- **SciPy**: For scientific computing and advanced mathematical functions.
- **lmfit**: For curve fitting and optimization.
- **Jupyter**: For interactive computing and data exploration.
- **nbformat**: For Jupyter notebook formatting.
- **pyarrow**: For reading and writing data in various formats.

For convenience run these commands to install the packages:

```bash
conda install -c conda-forge pandas numpy==2.2 plotly plotly-resampler holoviews scipy lmfit jupyter nbformat pyarrow datashader
pip install kaleido
```

Or using `uv` (after creating the environment):

```bash
uv add jupyter holoviews datashader plotly plotly-resampler lmfit altair vegafusion fastparquet
``` 

## Columns

The reading is done from a CSV file which is output from TrackMate (ImageJ plugin).
The typical column names are separated by underscores which I removed for simplicity.
Some prefixes are also removed to make the column names more readable.
For example, `TRACK_ID` becomes `TrackID`, `POSITION_X` becomes `X`, etc.

The other columns are added after each analysis is applied to the data.

- **TrackID**: Identifier for the track within the file
- **X**: X-coordinate of the particle's position
- **Y**: Y-coordinate of the particle's position
- **T**: Time point of the measurement
- **FileID**: Identifier for the file from which the trajectory was extracted
- **UID**: Unique identifier for each trajectory
- **MSD**: Mean Squared Displacement
- **Lag'T**: Lag time between cumulative frames
- **D_Norm**: Diffusion coefficient in a normal diffusion model
- **D_Norm_Error**: Error in the diffusion coefficient for the normal diffusion model
- **D_Anom**: Diffusion coefficient in an anomalous diffusion model
- **a_Anom**: Anomalous diffusion exponent in the anomalous diffusion model
- **JD_Freq**: Frequency count of jumps in the Jump Distance histogram
- **JD_Bin_Center**: Center of the bin for the Jump Distance histogram
- **JD1x_D**: Diffusion coefficient for anomalous diffusion model with one component
- **JD1x_Alpha**: Anomalous diffusion exponent for anomalous diffusion model with one component
- **JD1xn_D**: Diffusion coefficient for normal diffusion model with one component
- **JD2x_D1**: Diffusion coefficient for anomalous diffusion model with two components
- **JD2x_D2**: Diffusion coefficient for anomalous diffusion model with two components
- **JD2x_Alpha1**: Anomalous diffusion exponent for anomalous diffusion model with two components
- **JD2x_Alpha2**: Anomalous diffusion exponent for anomalous diffusion model with two components
- **JD2x_a1**: First Amplitude for the first component in the anomalous diffusion model with two components
- **JD2x_a2**: Second Amplitude for the second component in the anomalous diffusion model with two components
- **JD2xn_D1**: Diffusion coefficient for normal diffusion model with two components
- **JD2xn_D2**: Diffusion coefficient for normal diffusion model with two components
- **JD2xn_a1**: First Amplitude for the first component in the normal diffusion model with two components
- **JD2xn_a2**: Second Amplitude for the second component in the normal diffusion model with two components
- **JD3x_D1**: Diffusion coefficient for anomalous diffusion model with three components
- **JD3x_D2**: Diffusion coefficient for anomalous diffusion model with three components
- **JD3x_D3**: Diffusion coefficient for anomalous diffusion model with three components
- **JD3x_Alpha1**: Anomalous diffusion exponent for anomalous diffusion model with three components
- **JD3x_Alpha2**: Anomalous diffusion exponent for anomalous diffusion model with three components
- **JD3x_Alpha3**: Anomalous diffusion exponent for anomalous diffusion model with three components
- **JD3x_a1**: First Amplitude for the first component in the anomalous diffusion model with three components
- **JD3x_a2**: Second Amplitude for the second component in the anomalous diffusion model with three components
- **JD3x_a3**: Third Amplitude for the third component in the anomalous diffusion model with three components
- **JD3xn_D1**: Diffusion coefficient for normal diffusion model with three components
- **JD3xn_D2**: Diffusion coefficient for normal diffusion model with three components
- **JD3xn_D3**: Diffusion coefficient for normal diffusion model with three components
- **JD3xn_a1**: First Amplitude for the first component in the normal diffusion model with three components
- **JD3xn_a2**: Second Amplitude for the second component in the normal diffusion model with three components
- **JD3xn_a3**: Third Amplitude for the third component in the normal diffusion model with three components
- **Alpha_Flag_THS**: Flags the alpha value from a single point in the normalized MSD plot (in log-log scale)
- **Alpha_Flag_Fit**: Flags the alpha value by fitting the normalized MSD plot to a line (in log-log scale)
- **Alpha**: Anomalous diffusion exponent from the normalized MSD plot fitting (in log-log scale)
- **D_Mean_Alpha**: Diffusion coefficient in an anomalous diffusion model where alpha is fixed to mean value of its class (e.g., 1.0 for normal diffusion)
- **D_Mean_Alpha_Error**: Error of D_Mean_Alpha
- **D_Fixed_Alpha**: Diffusion coefficient in an anomalous diffusion model where alpha is fixed to a specific value from the log-log normalized MSD plot
- **D_Fixed_Alpha_Error**: Error of D_Fixed_Alpha
- **Conf_Level**: The level that determines the probability of particle undergoing nonrandom diffusion. This behavior is related to confinement domains.
- **Conf_Label**: *To be implemented* Label for the confinement level, which can be 'confined', 'not-confined'.
- **MW_D**: Diffusion coefficient from the transient Mean Squared Displacement (MSD) analysis for a moving window.
- **MW_D_Error**: Error of MW_D.
