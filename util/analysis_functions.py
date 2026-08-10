"""
All the analysis functions are defined here.
"""
import numpy as np
import pandas as pd
import param  # noqa: F401
from lmfit import Model, Parameters
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist

import util.constants as const


def calculate_msd(df) -> pd.DataFrame:
    """
    Calculate the Mean Squared Displacement (MSD) for a single particle trajectory.

    This function computes the MSD for various time lags. To account for missing frames 
    in the trajectory, it uses a frame-to-index mapping to ensure that displacements 
    are calculated between points separated by the exact physical time lag, 
    rather than simply by row index.

    The maximum lag time is limited to a percentage of the total trajectory length 
    (defined by `const.MSD_LENGTH_DIVISOR`) to ensure statistical reliability, 
    as the number of available pairs decreases as lag increases.

    Parameters:
        df (pd.DataFrame): DataFrame containing the trajectory data. 
                           Must include columns:
                           - 'Frame': The frame number (used to calculate actual time lag).
                           - 'X': X-coordinate of the particle.
                           - 'Y': Y-coordinate of the particle.

    Returns:
        pd.DataFrame: The original DataFrame with two additional columns:
                      - 'MSD': The calculated Mean Squared Displacement for that lag.
                      - 'Lag_T': The physical time lag in seconds.
    """
    # ... (keep existing docstring and initial variables)
    dt = const.DT
    cf = const.NANOMETER_TO_MICROMETER 
    max_lag = round(const.MSD_LENGTH_DIVISOR * len(df))
   
    frame_map = {frame: idx for idx, frame in enumerate(df['Frame'])}
    frames = df['Frame'].values
    x = df['X'].values
    y = df['Y'].values

    msd_results = {}
    for lag in range(1, max_lag + 1):
        displacements = []
        for i in range(len(df)):
            target_frame = frames[i] + lag
            if target_frame in frame_map:
                j = frame_map[target_frame]
                # Calculate squared displacement and convert to microns
                dist_sq = ((x[j] - x[i]) * cf)**2 + ((y[j] - y[i]) * cf)**2
                displacements.append(dist_sq)
        
        if displacements:
            msd_results[lag] = np.mean(displacements)
        else:
            msd_results[lag] = np.nan
        # ------------------------------------------

    # ... (keep the rest of the function as is)
    msd_df = pd.DataFrame(list(msd_results.items()), columns=['Lag_T', 'MSD'])
    msd_df['Lag_T'] = msd_df['Lag_T'] * dt
    
    df = df.reset_index(drop=True)
    df["MSD"] = msd_df["MSD"].reset_index(drop=True)
    df["Lag_T"] = msd_df["Lag_T"].reset_index(drop=True)
    
    return df

def calculate_msd_old(df) -> pd.DataFrame:
    """
    Calculate the Mean Squared Displacement (MSD) for a given DataFrame.
    The maximum lag time is set to 60% of the total time. If its more than this, 
    the calculated MSD will be inaccurate. Simply there will be not 
    enough data to calculate the MSD.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing the trajectory data with columns 'X' and 'Y'.
        dt (float): Time interval between frames.
    Returns:
        pandas.DataFrame: DataFrame containing the MSD results with columns 'Lag', 'MSD', and 'T'.
    """
    dt = const.DT
    cf = const.NANOMETER_TO_MICROMETER  # Conversion factor from nanometers to micrometers
    max_lag =round(const.MSD_LENGTH_DIVISOR * len(df))  # Set the maximum lag time to 60% of the total time
    msd_results = {}
    for lag in range(1, max_lag + 1):
        dy = df['Y'].diff(periods=lag).dropna()
        dx = df['X'].diff(periods=lag).dropna()
        displacement = (dx*cf)**2 + (dy*cf)**2 # convert to microns
        msd_results[lag] = displacement.mean()
    msd_df = pd.DataFrame(list(msd_results.items()), columns=['Lag_T', 'MSD'])
    msd_df['Lag_T'] = msd_df['Lag_T'] * dt
    
    df = df.reset_index(drop=True)
    df["MSD"] = msd_df["MSD"].reset_index(drop=True)
    df["Lag_T"] = msd_df["Lag_T"].reset_index(drop=True)
    
    return df

def remove_msd_offset_with_lr(
        df: pd.DataFrame, 
        localization_precision: float = const.LOCALIZATION_PRECISION_UM, 
        replace: bool = True) -> pd.DataFrame:
    """
    Removes the offset from the MSD data using linear regression.
    The offset is calculated as the y-intercept of the linear regression line fitted to the first few points of the MSD vs. Lag_T data.
    This offset is then subtracted from the entire MSD column, and the result is stored in a new 'MSD_NoOff' column.

    Args:
        df (pd.DataFrame): DataFrame containing a single trajectory with 'MSD' and 'Lag_T' columns.
        localization_precision (float): The localization precision in micrometers. 
                                         This value is used to determine the number of points to use for fitting.
        replace (bool): If True, the original 'MSD' column will be replaced with the offset-corrected values.
    Returns:
        pd.DataFrame: The input DataFrame with an additional 'MSD_NoOff' column.
                      If the trajectory has fewer than the required points for fitting (currently 4), 'MSD_NoOff' will be NaN for that trajectory.
    """

    sig = localization_precision

    if replace:
        df['MSD'] = df['MSD'] - 2*sig**2
    else:
        df['MSD_NoOff'] = df['MSD'] - 2*sig**2

    return df


def remove_msd_offset(df_trajectory: pd.DataFrame, replace: bool = True) -> pd.DataFrame:
    """
    Fits the first few points of a trajectory's MSD vs. Lag_T data to a line
    to determine a y-intercept (offset), and then subtracts this offset from
    the entire MSD column, storing the result in a new 'MSD_NoOff' column.

    **CAUSION**: finding offset from fitting is not reliable. 
    Refer to Michalet 2010 (Mean square displacement analysis of single-particle trajectories with localization error, Phys. Rev. E 82, 041914)
    
    This can be useful to correct for localization error bias in MSD measurements.

    Args:
        df_trajectory (pd.DataFrame): DataFrame containing a single trajectory
                                      with 'MSD' and 'Lag_T' columns.

    Returns:
        pd.DataFrame: The input DataFrame with an additional 'MSD_NoOff' column.
                      If the trajectory has fewer than the required points for fitting
                      (currently 4), 'MSD_NoOff' will be NaN for that trajectory.
    """
    num_points_for_fit = 4 # As requested by the user

    if len(df_trajectory) < num_points_for_fit:
        df_trajectory['MSD_NoOff'] = np.nan
        return df_trajectory

    # Get the first `num_points_for_fit` data points
    x_fit = df_trajectory['Lag_T'].iloc[:num_points_for_fit].values
    y_fit = df_trajectory['MSD'].iloc[:num_points_for_fit].values

    # Fit a straight line (degree 1 polynomial) to these points
    # polyfit returns coefficients [slope, intercept]
    try:
        coefficients = np.polyfit(x_fit, y_fit, 1)
        offset = coefficients[1] # The y-intercept is the second coefficient
    except Exception:
        # Handle cases where fitting might fail (e.g., all y_fit are NaN or non-finite)
        offset = np.nan

    # Subtract the offset from the entire MSD column to create 'MSD_NoOff'
    if replace:
        df_trajectory['MSD'] = df_trajectory['MSD'] - offset
    else:
        df_trajectory['MSD_NoOff'] = df_trajectory['MSD'] - offset
    
    return df_trajectory


# MARK: Diffusion Coefficient Calculation
def normal_diffusion_msd(t, d):
    """
    Normal diffusion in 2D.
    Parameters:
        t (float): Time lag.
        d (float): Diffusion coefficient.
    Returns:
        float: The mean squared displacement for the given time lag and diffusion coefficient.
    """
    return 4 * d * t

def anomalous_diffusion_msd(t, d, a):
    """
    Anomalous diffusion in 2D.
    Parameters:
        t (float): Time lag.
        d (float): Diffusion coefficient.
        a (float): Anomalous exponent.
    Returns:
        float: The mean squared displacement for the given time lag and diffusion coefficient.
    """
    return 4 * d * t**a

def calculate_diff_d(df) -> pd.DataFrame:
    """
    Calculate the normal diffusion coefficient for a given DataFrame.
    The **MSD** and **Lag_T** columns are used to calculate the diffusion coefficient.
    
    Parameters:
        df["MSD"] (float): Mean Squared Displacement.
        df["Lag_T"] (float): Lag time.
    Returns:
        pandas.DataFrame: DataFrame containing the diffusion coefficient and its error.
        Two new columns are added to the DataFrame: **D_Norm** and **D_Norm_error**.
    """
    # Check if the DataFrame is empty. I don't know why this is necessary but curve_fit
    # throws an error that ydata is empty if this is not done.
    if len(df) == 0:
        return df

    ydata = df["MSD"].dropna().values
    xdata = df["Lag_T"].dropna().values

    # fit the MSD to a line
    popt, pcov = curve_fit(normal_diffusion_msd, xdata, ydata)
    d_coefficient = popt[0]
    d_error = np.sqrt(np.diag(pcov))[0]
    # add the d coefficient to the dataframe
    df["D_Norm"] = d_coefficient
    df["D_Norm_error"] = d_error

    return df

def calculate_anom_diff_coef(df) -> pd.DataFrame:
    """
    Calculate the anomalous diffusion coefficient and anomalous exponent for a given DataFrame.
    **MSD** and **Lag_T** columns are used to calculate the diffusion coefficient and 
    anomalous exponent.

    Parameters:
        df["MSD"] (float): Mean Squared Displacement.
        df["Lag_T"] (float): Lag time.
    Returns:
        pandas.DataFrame: DataFrame containing the anomalous diffusion parameters.
        Two new columns are added to the DataFrame: **D_Anom** and **a_Anom**.
        
    """
    # Check if the DataFrame is empty. I don't know why this is necessary but curve_fit
    # throws an error that ydata is empty if this is not done.
    if len(df) == 0:
        return df

    ydata = df["MSD"].dropna().values
    xdata = df["Lag_T"].dropna().values

    # fit the MSD to a line
    popt, pcov = curve_fit(anomalous_diffusion_msd, xdata, ydata)

    # add the d coefficient to the dataframe
    df["D_Anom"] = popt[0]
    df["a_Anom"] = popt[1]

    return df


# Jump Distance Ananlysis
def jd_exp(x, a, b):
    """
    Exponential function for Jump Distance Fitting.
    Parameters:
        x (float): Input value.
        a (float): Diffusion Coefficient.
        b (float): Anomalouseness exponent.
    Returns:
        float: The value of the exponential function for the given input.
    """
    dt = const.DT
    return np.exp(- x**2 / (4 * a * dt**b))

def jd_1exp(x:float, a:float, b:float)->float:
    """
    Single exponential Jump Distance model.
    Parameters:
        x (float): Input value.
        a (float): Amplitude.
        b (float): Diffusion Coefficient.
        c (float): Anomalousness exponent.
    Returns:
        float: The value of the single exponential function for the given input.
    """
    return 1 - jd_exp(x, a, b)

def jd_1exp_norm(x, a):
    """
    Normalized single exponential Jump Distance model.
    Parameters:
        x (float): Input value.
        a (float): Amplitude.
    Returns:
        float: The value of the normalized single exponential function for the given input.
    """
    return 1 - jd_exp(x, a, 1)

def jd_2exp(x, a1, a2, b1, b2, c1, c2):
    """
    Double exponential Jump Distance model.
    Parameters:
        x (float): Input value.
        a1 (float): Amplitude of the first exponential.
        a2 (float): Amplitude of the second exponential.
        b1 (float): Diffusion Coefficient of the first exponential.
        b2 (float): Diffusion Coefficient of the second exponential.
        c1 (float): Anomalousness exponent of the first exponential.
        c2 (float): Anomalousness exponent of the second exponential.
    Returns:
        float: The value of the double exponential function for the given input.
        This function is used to fit the Jump Distance data to a double exponential model.
    """
    return 1 - a1 * jd_exp(x, b1, c1) - a2 * jd_exp(x, b2, c2)

def jd_2exp_norm(x, a1, a2, b1, b2):
    """
    Doube exponential Jump Distance model in normal diffusion.
    Parameters:
        x (float): Input value.
        a1 (float): Amplitude of the first exponential.
        a2 (float): Amplitude of the second exponential.
        b1 (float): Diffusion Coefficient of the first exponential.
        b2 (float): Diffusion Coefficient of the second exponential.
    Returns:
        float: The value of the double exponential function for the given input.
        This function is used to fit the Jump Distance data to a double exponential model.
    """
    return 1 - a1 * jd_exp(x, b1, 1) - a2 * jd_exp(x, b2, 1)

def jd_3exp(x, a1, a2, a3, b1, b2, b3, c1, c2, c3):
    """
    Triple exponential Jump Distance model.
    Parameters:
        x (float): Input value.
        a1 (float): Amplitude of the first exponential.
        a2 (float): Amplitude of the second exponential.
        a3 (float): Amplitude of the third exponential.
        b1 (float): Diffusion Coefficient of the first exponential.
        b2 (float): Diffusion Coefficient of the second exponential.
        b3 (float): Diffusion Coefficient of the third exponential.
        c1 (float): Anomalousness exponent of the first exponential.
        c2 (float): Anomalousness exponent of the second exponential.
        c3 (float): Anomalousness exponent of the third exponential.
    Returns:
        float: The value of the triple exponential function for the given input.
        This function is used to fit the Jump Distance data to a triple exponential model.
    """
    return 1 - a1 * jd_exp(x, b1, c1) - a2 * jd_exp(x, b2, c2) - a3 * jd_exp(x, b3, c3)

def jd_3exp_norm(x, a1, a2, a3, b1, b2, b3):
    """
    Triple exponential Jump Distance model in normal diffusion.
    Parameters:
        x (float): Input value.
        a1 (float): Amplitude of the first exponential.
        a2 (float): Amplitude of the second exponential.
        a3 (float): Amplitude of the third exponential.
        b1 (float): Diffusion Coefficient of the first exponential.
        b2 (float): Diffusion Coefficient of the second exponential.
        b3 (float): Diffusion Coefficient of the third exponential.
    Returns:
        float: The value of the triple exponential function for the given input.
        This function is used to fit the Jump Distance data to a triple exponential model.
    """
    return 1 - a1 * jd_exp(x, b1, 1) - a2 * jd_exp(x, b2, 1) - a3 * jd_exp(x, b3, 1)

def calculate_jd(df, bin_size=0.02):
    """
    Calculate the Jump Distance (JD) for a given DataFrame.

    The JD is calculated as the cumulative sum of the distances between consecutive points.
    The distances are calculated using the Euclidean distance formula.
    The distances are then binned into intervals of size `bin_size`.
    The cumulative sum of the histogram is normalized to create a cumulative distribution function (CDF).
    Parameters:
        df (pandas.DataFrame): DataFrame containing the trajectory data with columns **X** and **Y**.
        bin_size (float): Size of the bins for the histogram.
    Returns:
        pandas.DataFrame: DataFrame containing the JD results with columns **JD_Freq** and **JD_Bin_Center**.
    """
    dist = np.sqrt((df['X'] - df['X'].shift())**2 + (df['Y'] - df['Y'].shift())**2)
    dist = dist.fillna(0)  # Fill NaN values with 0 for the first element
    dist = dist/1000  # convert to microns
    # Build histogram of distances
    bins = np.arange(dist.min(), dist.max() + bin_size, bin_size)
    hist = np.histogram(dist, bins=bins)
    hist_cumsum = np.cumsum(hist[0])
    bin_edges = hist[1]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist_cumsum_norm = hist_cumsum / hist_cumsum[-1]
    jd_df = pd.DataFrame(hist_cumsum_norm, columns=['JD_Freq'])
    jd_df['JD_Bin_Center'] = bin_centers

    df = df.reset_index(drop=True)
    df['JD_Freq'] = jd_df['JD_Freq'].reset_index(drop=True)
    df['JD_Bin_Center'] = jd_df['JD_Bin_Center'].reset_index(drop=True)
    return df

def fit_jd_1exp(df):
    """
    Fit the Jump Distance data to a single exponential model with D and alpha.

    The dataframe should already contain **JD_Freq** and **JD_Bin_Center** columns.
    This function fits the Jump Distance data to a single exponential model.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing the Jump Distance 
            data with columns **JD_Freq** and **JD_Bin_Center**.
    Returns:
        pandas.DataFrame: DataFrame containing the fitted parameters and their errors.
        It will add the new columns **JD1x_D**, **JD1x_D_error**,
        **JD1x_Alpha**, and **JD1x_Alpha_error** to the DataFrame.
    """
    # Check if the DataFrame is empty. I don't know why this is necessary but curve_fit
    # throws an error that ydata is empty if this is not done.
    if len(df) == 0:
        return df

    ydata = df["JD_Freq"].dropna().values
    xdata = df["JD_Bin_Center"].dropna().values

    # fit the MSD to a line
    p0 = [1, 1]  # Initial guess for the parameters
    popt, pcov = curve_fit(jd_1exp, xdata, ydata, p0=p0, bounds=(0, np.inf), method='trf')
    
    # add the d coefficient to the dataframe
    df["JD1x_D"] = popt[0]
    df["JD1x_D_error"] = np.sqrt(np.diag(pcov))[0]
    df["JD1x_Alpha"] = popt[1]
    df["JD1x_Alpha_error"] = np.sqrt(np.diag(pcov))[1]
    return df

def fit_jd_1exp_norm(df):
    """
    Fit the Jump Distance data to a single exponential model for normal diffusion.
    Parameters:
        df (pandas.DataFrame): DataFrame containing the Jump Distance 
            data with columns **JD_Freq** and **JD_Bin_Center**.
    Returns:
        pandas.DataFrame: DataFrame containing the fitted parameters and their errors.
            It will add the new columns **JD1xn_D** and **JD1xn_D_error** to the DataFrame.
    """
    # Check if the DataFrame is empty. I don't know why this is necessary but curve_fit
    # throws an error that ydata is empty if this is not done.
    if len(df) == 0:
        return df

    ydata = df["JD_Freq"].dropna().values
    xdata = df["JD_Bin_Center"].dropna().values
    
    # fit the MSD to a line
    p0 = [1]  # Initial guess for the parameters
    popt, pcov = curve_fit(jd_1exp_norm, xdata, ydata, p0=p0, bounds=(0, np.inf))
    
    # add the d coefficient to the dataframe
    df["JD1xn_D"] = popt[0]
    df["JD1xn_D_error"] = np.sqrt(np.diag(pcov))[0]
    return df

def fit_jd_2exp(df):
    """
    Fit the Jump Distance data to a double exponential model.
    Parameters:
        df (pandas.DataFrame): DataFrame containing the Jump 
            Distance data with columns 'JD_Freq' and 'JD_Bin_Center'.
    Returns:
        pandas.DataFrame: DataFrame containing the fitted parameters and their errors.
        It will add the new columns **JD2x_a1**, **JD2x_a1_error**,
        **JD2x_a2**, **JD2x_a2_error**, **JD2x_D1**, **JD2x_D1_error**,
        **JD2x_D2**, **JD2x_D2_error**, **JD2x_Alpha1**, **JD2x_Alpha1_error**,
        **JD2x_Alpha2**, and **JD2x_Alpha2_error**
    """
    # Check if the DataFrame is empty. I don't know why this is necessary but curve_fit
    # throws an error that ydata is empty if this is not done.
    if len(df) == 0:
        return df

    ydata = df["JD_Freq"].dropna().values
    xdata = df["JD_Bin_Center"].dropna().values
    
    model = Model(jd_2exp, independent_vars=['x'])
    params = Parameters()
    params.add('a1', value=0.5, min=0, max=1)
    params.add('a2', value=0.5, min=0, max=1)
    params['a2'].expr = '1 - a1'
    params.add('b1', value=1, min=0, max=10)
    params.add('b2', value=1, min=0, max=10)
    params.add('c1', value=1, min=0, max=10)
    params.add('c2', value=1, min=0, max=10)
    params.add('sum_a', expr='a1 + a2 - 1')
    
    results = model.fit(ydata, params, x=xdata, method='leastsq')
    
    # add the d coefficient to the dataframe
    df["JD2x_a1"] = results.params['a1'].value
    df["JD2x_a1_error"] = results.params['a1'].stderr
    df["JD2x_a2"] = results.params['a2'].value
    df["JD2x_a2_error"] = results.params['a2'].stderr
    df["JD2x_D1"] = results.params['b1'].value
    df["JD2x_D1_error"] = results.params['b1'].stderr
    df["JD2x_D2"] = results.params['b2'].value
    df["JD2x_D2_error"] = results.params['b2'].stderr
    df["JD2x_Alpha1"] = results.params['c1'].value
    df["JD2x_Alpha1_error"] = results.params['c1'].stderr
    df["JD2x_Alpha2"] = results.params['c2'].value
    df["JD2x_Alpha2_error"] = results.params['c2'].stderr
    return df

def fit_jd_2exp_norm(df):
    """
    Fit the Jump Distance data to a double exponential model for normal diffusion.
    Parameters:
        df (pandas.DataFrame): DataFrame containing the Jump
            Distance data with columns **JD_Freq** and **JD_Bin_Center**.
    Returns:
        pandas.DataFrame: DataFrame containing the fitted parameters and their errors.
            It will add the new columns **JD2xn_a1**, **JD2xn_a1_error**,
            **JD2xn_a2**, **JD2xn_a2_error**, **JD2xn_D1**, **JD2xn_D1_error**,
            **JD2xn_D2**, and **JD2xn_D2_error**.
    """
    # Check if the DataFrame is empty. I don't know why this is necessary but curve_fit
    # throws an error that ydata is empty if this is not done.
    if len(df) == 0:
        return df

    ydata = df["JD_Freq"].dropna().values
    xdata = df["JD_Bin_Center"].dropna().values

    model = Model(jd_2exp_norm, independent_vars=['x'])

    params = Parameters()
    params.add('a1', value=0.5, min=0, max=1)
    params.add('a2', value=0.5, min=0, max=1)
    params['a2'].expr = '1 - a1'
    params.add('b1', value=1, min=0, max=10)
    params.add('b2', value=1, min=0, max=10)
    params.add('sum_a', expr='a1 + a2 - 1')

    results = model.fit(ydata, params, x=xdata, method='leastsq')
    
    # add the d coefficient to the dataframe
    df["JD2xn_a1"] = results.params['a1'].value
    df["JD2xn_a1_error"] = results.params['a1'].stderr
    df["JD2xn_a2"] = results.params['a2'].value
    df["JD2xn_a2_error"] = results.params['a2'].stderr
    df["JD2xn_D1"] = results.params['b1'].value
    df["JD2xn_D1_error"] = results.params['b1'].stderr
    df["JD2xn_D2"] = results.params['b2'].value
    df["JD2xn_D2_error"] = results.params['b2'].stderr
    return df

def fit_jd_3exp(df):
    """
    Fit the Jump Distance data to a three exponential model.
    Parameters:
        df (pandas.DataFrame): DataFrame containing the Jump
            Distance data with columns **JD_Freq** and **JD_Bin_Center**.
    Returns:
        pandas.DataFrame: DataFrame containing the fitted parameters and their errors.
        It will add the new columns **JD3x_a1**, **JD3x_a1_error**,
        **JD3x_a2**, **JD3x_a2_error**, **JD3x_a3**, **JD3x_a3_error**,
        **JD3x_D1**, **JD3x_D1_error**, **JD3x_D2**, **JD3x_D2_error**,
        **JD3x_D3**, **JD3x_D3_error**, **JD3x_Alpha1**, **JD3x_Alpha1_error**,
        **JD3x_Alpha2**, **JD3x_Alpha2_error**, **JD3x_Alpha3**, and **JD3x_Alpha3_error**.
    """
    # Check if the DataFrame is empty. I don't know why this is necessary but curve_fit
    # throws an error that ydata is empty if this is not done.
    if len(df) == 0:
        return df

    ydata = df["JD_Freq"].dropna().values
    xdata = df["JD_Bin_Center"].dropna().values

    # To ensure length of the data is greater than the number of parameters
    if len(xdata) < 10:
        return df
    
    model = Model(jd_3exp, independent_vars=['x'])
    params = Parameters()
    params.add('a1', value=0.33, min=0, max=1)
    params.add('a2', value=0.33, min=0, max=1)
    params.add('a3', value=0.33, min=0, max=1)
    params.add('b1', value=1, min=0, max=10)
    params.add('b2', value=1, min=0, max=10)
    params.add('b3', value=1, min=0, max=10)
    params.add('c1', value=1, min=0, max=10)
    params.add('c2', value=1, min=0, max=10)
    params.add('c3', value=1, min=0, max=10)
    params.add('sum_a', expr='a1 + a2 + a3 - 1')
    
    results = model.fit(ydata, params, x=xdata, method='leastsq')
    
    # add the d coefficient to the dataframe
    df["JD3x_a1"] = results.params['a1'].value
    df["JD3x_a1_error"] = results.params['a1'].stderr
    df["JD3x_a2"] = results.params['a2'].value
    df["JD3x_a2_error"] = results.params['a2'].stderr
    df["JD3x_a3"] = results.params['a3'].value
    df["JD3x_a3_error"] = results.params['a3'].stderr
    df["JD3x_D1"] = results.params['b1'].value
    df["JD3x_D1_error"] = results.params['b1'].stderr
    df["JD3x_D2"] = results.params['b2'].value
    df["JD3x_D2_error"] = results.params['b2'].stderr
    df["JD3x_D3"] = results.params['b3'].value
    df["JD3x_D3_error"] = results.params['b3'].stderr
    df["JD3x_Alpha1"] = results.params['c1'].value
    df["JD3x_Alpha1_error"] = results.params['c1'].stderr
    df["JD3x_Alpha2"] = results.params['c2'].value
    df["JD3x_Alpha2_error"] = results.params['c2'].stderr
    df["JD3x_Alpha3"] = results.params['c3'].value
    df["JD3x_Alpha3_error"] = results.params['c3'].stderr
    return df

def fit_jd_3exp_norm(df):
    """
    Fit the Jump Distance data to a three exponential model for normal diffusion.
    Parameters:
        df (pandas.DataFrame): DataFrame containing the Jump
            Distance data with columns **JD_Freq** and **JD_Bin_Center**.
    Returns:
        pandas.DataFrame: DataFrame containing the fitted parameters and their errors.
        It will add the new columns **JD3xn_a1**, **JD3xn_a1_error**,
        **JD3xn_a2**, **JD3xn_a2_error**, **JD3xn_a3**, **JD3xn_a3_error**,
        **JD3xn_D1**, **JD3xn_D1_error**, **JD3xn_D2**, **JD3xn_D2_error**,
        **JD3xn_D3**, **JD3xn_D3_error**, **JD3xn_Alpha1**, **JD3xn_Alpha1_error**,
        **JD3xn_Alpha2**, **JD3xn_Alpha2_error**, **JD3xn_Alpha3**, and **JD3xn_Alpha3_error**.
    """
    # Check if the DataFrame is empty. I don't know why this is necessary but curve_fit
    # throws an error that ydata is empty if this is not done.
    if len(df) == 0:
        return df

    ydata = df["JD_Freq"].dropna().values
    xdata = df["JD_Bin_Center"].dropna().values
    
    # To ensure length of the data is greater than the number of parameters
    if len(xdata) < 10:
        return df
    
    model = Model(jd_3exp_norm, independent_vars=['x'])
    
    params = Parameters()
    params.add('a1', value=0.33, min=0, max=1)
    params.add('a2', value=0.33, min=0, max=1)
    params.add('a3', value=0.33, min=0, max=1)
    params.add('b1', value=1, min=0, max=10)
    params.add('b2', value=1, min=0, max=10)
    params.add('b3', value=1, min=0, max=10)
    params.add('sum_a', expr='a1 + a2 + a3 - 1')
    
    results = model.fit(ydata, params, x=xdata, method='leastsq')
    
    # add the d coefficient to the dataframe
    df["JD3xn_a1"] = results.params['a1'].value
    df["JD3xn_a1_error"] = results.params['a1'].stderr
    df["JD3xn_a2"] = results.params['a2'].value
    df["JD3xn_a2_error"] = results.params['a2'].stderr
    df["JD3xn_a3"] = results.params['a3'].value
    df["JD3xn_a3_error"] = results.params['a3'].stderr
    df["JD3xn_D1"] = results.params['b1'].value
    df["JD3xn_D1_error"] = results.params['b1'].stderr
    df["JD3xn_D2"] = results.params['b2'].value
    df["JD3xn_D2_error"] = results.params['b2'].stderr
    df["JD3xn_D3"] = results.params['b3'].value
    df["JD3xn_D3_error"] = results.params['b3'].stderr
    return df

def flag_alpha_by_val(df):
    """
    Flag the anomalousness exponent (Alpha) from the normalized 
    MSD plot in log-log scale from one of the first few points.
    This function calculates the normalized MSD value at a specific point
    (defined by `const.MSD_SLOPE_POINTS`) and classifies it based on the
    ALPHA_THRESHOLDS defined in the constants module.
    The classification is done into four categories: 'ignore', 'sub', 'sup', and 'normal'.
    Parameters:
        df (pandas.DataFrame): DataFrame containing the MSD data
            with columns **Lag_T** and **MSD**.
    Returns:
        pandas.DataFrame: DataFrame with an additional column 
            **Alpha_Flag_THS** indicating the anomalousness exponent.
    """
    alpha_ignore = const.ALPHA_THRESHOLDS['ignore']
    alpha_sub = const.ALPHA_THRESHOLDS['sub']
    alpha_sup = const.ALPHA_THRESHOLDS['sup']
    
    msd_points = const.MSD_SLOPE_POINTS
    df['Alpha_Flag_THS'] = 'normal'

    norm_msd = df['MSD'].iloc[msd_points-1] / df['MSD'].iloc[0]
    if norm_msd <= msd_points ** alpha_ignore:
        df['Alpha_Flag_THS'] = 'ignore'
    elif norm_msd <= msd_points ** alpha_sub:
        df['Alpha_Flag_THS'] = 'sub'
    elif norm_msd >= msd_points ** alpha_sup:
        df['Alpha_Flag_THS'] = 'sup'

    return df

def flag_alpha_by_fit(df):
    """
    Flag the anomalousness exponent (Alpha) from the normalized 
    MSD plot in log-log scale through fitting a line to the first few points.
    Parameters:
        df (pandas.DataFrame): DataFrame containing the MSD data 
            with columns **Lag_T** and **MSD**.
    Returns:
        pandas.DataFrame: DataFrame with an additional column 
        **Alpha_Flag_Fit** indicating the anomalousness exponent.
        **Alpha** column contains the slope of the line fitted to the log-log data.
    """
    num_points = const.MSD_SLOPE_POINTS
    df['Alpha_Flag_Fit'] = 'normal'
    df['Alpha'] = np.nan

    msd = df['MSD'].iloc[:num_points]
    lag_t = df['Lag_T'].iloc[:num_points]

    # Fit a straight line to log-log data
    log_lag_t = np.log(lag_t)
    log_msd = np.log(msd)
    slope, _ = np.polyfit(log_lag_t, log_msd, 1)

    # Flag based on slope thresholds
    alpha_ignore = const.ALPHA_THRESHOLDS['ignore']
    alpha_sub = const.ALPHA_THRESHOLDS['sub']
    alpha_sup = const.ALPHA_THRESHOLDS['sup']

    if slope <= alpha_ignore:
        df['Alpha_Flag_Fit'] = 'ignore'
    elif slope <= alpha_sub:
        df['Alpha_Flag_Fit'] = 'sub'
    elif slope >= alpha_sup:
        df['Alpha_Flag_Fit'] = 'sup'

    df['Alpha'] = slope
    return df

def alpha_classes(df):
    """
    Calculate the slope (alpha) between the first point and a specified point
    on the Mean Squared Displacement (MSD) curve and classify it based on
    the 'Alpha_Flag' column.
    This function computes the normalized MSD value at a specific point
    (defined by `const.MSD_SLOPE_POINTS`) and calculates the slope (alpha)
    using the logarithmic ratio. The result is returned as a DataFrame
    containing the calculated alpha value and its corresponding class flag.
    Parameters:
        df (pd.DataFrame): Input DataFrame containing the MSD values and
            'Alpha_Flag' column.
    Returns:
        pd.DataFrame: A new DataFrame with two columns:
            - 'Alpha_Flag': The class flag for the alpha value.
            - 'Alpha': The calculated slope (alpha) value.
    Notes:
        - The slope is calculated between the first point and the point
        specified by `const.MSD_SLOPE_POINTS` on the MSD curve.
        - The 'Alpha_Flag' column is used to group and classify the alpha values.
    """

    result = pd.DataFrame()
    norm_msd_point = df['MSD'].iloc[const.MSD_SLOPE_POINTS-1] / df['MSD'].iloc[0]
    slope = np.log(norm_msd_point) / np.log(const.MSD_SLOPE_POINTS)
    result['Alpha_Flag'] = [df['Alpha_Flag_Fit'].iloc[0]]
    result['Alpha'] = [slope]
    return result

def calc_d_mean_alpha(df):
    """
    Calculate the mean diffusion coefficient (D) for each alpha class.
    Parameters:
        df (pd.DataFrame): Input DataFrame containing the diffusion coefficients and alpha classes.
    Returns:
        pd.DataFrame: A DataFrame with the mean diffusion coefficient for each alpha class.
        two columns are returned:
            - D_Mean_Alpha: The diffusion coefficient for an mean of alpha in each alpha class.
            - D_Mean_Alpha_error: The error associated with the diffusion coefficient.
    """
    alpha = df['Alpha_Mean'].iloc[0]

    def fixed_alpha_anom_diffusion_msd(t, d):
        """
        Calculate the mean squared displacement for anomalous diffusion in 2D
        with a fixed alpha value.
        Parameters:
            t (float): Time lag.
            d (float): Diffusion coefficient.
        Returns:
            float: The mean squared displacement for the given time lag and diffusion coefficient.
        """
        return 4 * d * t**alpha

    # Check if the DataFrame is empty. I don't know why this is necessary but curve_fit
    # throws an error that ydata is empty if this is not done.
    if len(df) == 0:
        return df

    ydata = df["MSD"].dropna().values
    xdata = df["Lag_T"].dropna().values

    # fit the MSD to a line
    popt, pcov = curve_fit(fixed_alpha_anom_diffusion_msd, xdata, ydata)
    d_coefficient = popt[0]
    d_error = np.sqrt(np.diag(pcov))[0]
    # add the d coefficient to the dataframe
    df["D_Mean_Alpha"] = d_coefficient
    df["D_Mean_Alpha_error"] = d_error

    return df

def calc_d_fix_alpha(df):
    """
    Calculate the diffusion coefficient (D) while keeping the alpha fixed from 
    the 'Alpha' column.
    Parameters:
        df (pd.DataFrame): Input DataFrame containing the diffusion coefficients and alpha classes.
    Returns:
        pd.DataFrame: A DataFrame with the diffusion coefficient for each alpha.
        two new columns are added:
            - D_Fixed_Alpha: The calculated diffusion coefficient with fixed alpha 
                from Alpha Column fitted to a normalized MSD
            - D_Fixed_Alpha_error: The error associated with the diffusion coefficient.
    """
    alpha = df['Alpha'].iloc[0]

    def fixed_alpha_anom_diffusion_msd(t, d):
        """
        Calculate the mean squared displacement for anomalous diffusion in 2D
        with a fixed alpha value.
        Parameters:
            t (float): Time lag.
            d (float): Diffusion coefficient.
        Returns:
            float: The mean squared displacement for the given time lag and diffusion coefficient.
        """
        return 4 * d * t**alpha

    # Check if the DataFrame is empty. I don't know why this is necessary but curve_fit
    # throws an error that ydata is empty if this is not done.
    if len(df) == 0:
        return df

    ydata = df["MSD"].dropna().values
    xdata = df["Lag_T"].dropna().values

    # fit the MSD to a line
    popt, pcov = curve_fit(fixed_alpha_anom_diffusion_msd, xdata, ydata)
    d_coefficient = popt[0]
    d_error = np.sqrt(np.diag(pcov))[0]
    # add the d coefficient to the dataframe
    df["D_Fixed_Alpha"] = d_coefficient
    df["D_Fixed_Alpha_error"] = d_error
    
    return df


def conf_calc_msd(lag_points, df):
    '''
    Calculate the Mean Squared Displacement (MSD) for the whole trajectory (not the segment).
    It calculates the MSD for each lag point from 1 to lag_points.
    Parameters:
        lag_points (int): The number of lag points to consider.
        df (pd.DataFrame): The DataFrame containing the trajectory data.
    Returns:
        dict: A dictionary with lag points as keys and their corresponding MSD values.
    '''
    msd = {}
    for lag in range(1, lag_points + 1):
        dy = df['Y'].diff(periods=lag).fillna(0)
        dx = df['X'].diff(periods=lag).fillna(0)
        msd[lag] = (dx**2 + dy**2).mean()
    return msd


def calc_confinement_level(df):
    """ Calculate the confinement level for each segment of the trajectory.
    This function computes the confinement level based on the diffusion coefficient
    and the maximum displacement of the segment.
    Parameters:
        df (pd.DataFrame): Input DataFrame containing the trajectory data with columns 'X' and 'Y'.
    Returns:
        pd.DataFrame: A DataFrame with an additional column 'Conf_Level' indicating the confinement level.
        The confinement level is calculated using the formula:
        Conf_Level = C1 - C2 * (D * t / R^2)
        t: time window determined by the window size of the segment and dt.
        R: maximum displacement in the segment.
        D: diffusion coefficient for the entire trajectory (not the segment).
    """
    df.reset_index(drop=True, inplace=True)
    dt = const.DT
    window_size = const.WINDOW_SIZE
    lag_points = const.MSD_FIT_POINTS
    c1 = const.CONF_C1
    c2 = const.CONF_C2
    tw = window_size * dt # Time window in seconds
    prob_thresh= const.PROBABILITY_THRESHOLD

    df['Conf_Level'] = np.nan  # Initialize the Confinement Level column

    msd = conf_calc_msd(lag_points, df)

    # Fitting to get diffusion coefficient for the segment
    ydata = pd.Series(msd).dropna().values
    xdata = np.arange(len(ydata)) * dt
    popt, pconv = curve_fit(normal_diffusion_msd, xdata, ydata)
    d = popt[0]  # Diffusion coefficient for the segment
    
    for i in range(len(df) - window_size + 1):
        segment = df.iloc[i:i + window_size]
        
        r = conf_calc_r(segment)
        
        log_prob = c1 - c2 * (d * tw / r**2)

        conf_level = -log_prob + np.log10(prob_thresh) if log_prob <= np.log10(prob_thresh) else 0

        df.iloc[i:i + window_size - 1, df.columns.get_loc('Conf_Level')] = conf_level


    return df


def conf_calc_r(segment):
    '''
    Calculate the maximum displacement (R) for a segment of the trajectory.
    The maximum displacement is calculated as the Euclidean distance from the first point in the segment.
    Parameters:
        segment (pd.DataFrame): The DataFrame containing the segment of the trajectory.
    Returns:
        float: The maximum displacement (R) for the segment.
    '''
    start_x, start_y = segment.iloc[0]['X'], segment.iloc[0]['Y']
    distances = np.sqrt((segment['X'] - start_x)**2 + (segment['Y'] - start_y)**2)
        # df.loc[i:i + window_size - 1, 'MW_R'] = distances.max()
    r= distances.max()
    return r

def label_confinement(df):
    '''
    TO BE IMPLEMENTED
    Label the points in a trajectory as confined or not confined based on the confinement level
    or other metrics.
    This function will add a new column 'Conf_Label' to the DataFrame, where each
    point is labeled as 'confined' or 'not confined'.
    '''
    return df

def calculate_diff_d_moving_window(df):
    """
    Calculate diffusion coeffiecients using a moving window approach.

    The moving windo is defined by the constants in the constants.py file.
    here is skipping length and window length. 
    Which are how many points to skip and how large is the window.
    and the diffusion coefficient is calculated
    Parameters:
        df (pandas.DataFrame): DataFrame containing the trajectory data with columns 'X' and 'Y'.
    Returns:
        pandas.DataFrame: DataFrame containing the diffusion 
        coefficient results with columns **MW_D** and **MW_D_error**.
    """

    # df is verified to be larger than window length during the mutation process
    # so no need to check here again.
    df.reset_index(drop=True, inplace=True)
    dt = const.DT
    window = const.TMSD_WINDOW_SIZE
    lag_points = const.TMSD_FIT_POINTS
    # create the columns for the diffusion coefficient and error and fill with NaN values
    df.loc[:, "MW_D"] = np.nan

    # Loop through the DataFrame with a moving window
    for i in range(len(df) - window + 1):
        # Select the window of data
        window_df = df.iloc[i : i + window]

        # Calculate the MSD for the window
        msd_results = {}
        for lag in range(1, lag_points + 1):
            dy = window_df["Y"].diff(periods=lag).dropna()
            dx = window_df["X"].diff(periods=lag).dropna()
            displacement_sqr = dx**2 + dy**2
            msd_results[lag] = displacement_sqr.mean()

        # Fit the MSD to a line and calculate the diffusion coefficient
        ydata = pd.Series(msd_results).dropna().values
        xdata = np.arange(len(ydata)) * dt

        try:
            popt, pcov = curve_fit(normal_diffusion_msd, xdata, ydata)
            d_coefficient = popt[0]
        except (ValueError, RuntimeError):
            d_coefficient = np.nan

        # Assign the diffusion coefficient to the corresponding rows in the DataFrame
        df.iloc[i : i + window - 1, df.columns.get_loc("MW_D")] = d_coefficient

    return df

def calculate_ensemble_msd(df, max_lag=20):
    """
    Calculate ensemble mean squared displacement for particle trajectories.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns ['Frame', 'X', 'Y', 'UID']
    max_lag (int): Maximum time lag to calculate MSD for
    
    Returns:
    pd.DataFrame: DataFrame with columns ['lag', 'msd', 'std_error', 'n_points']
    """
    
    # Sort dataframe by UID and Frame to ensure proper ordering
    df = df.sort_values(['UID', 'Frame']).reset_index(drop=True)
    
    # Initialize arrays to store MSD values for each lag
    msd_results = []
    
    # Calculate MSD for each time lag
    for lag in range(1, max_lag + 1):
        squared_displacements = []
        
        # Group by particle UID
        for uid, particle_data in df.groupby('UID'):
            # Sort by frame to ensure chronological order
            particle_data = particle_data.sort_values('Frame')
            
            # Get positions and frames
            frames = particle_data['Frame'].values
            x_positions = particle_data['X'].values
            y_positions = particle_data['Y'].values
            
            # Calculate squared displacements for this lag
            for i in range(len(frames) - lag):
                # Check if we have consecutive frames (or at least the required lag)
                if i + lag < len(frames):
                    # Calculate displacement
                    dx = x_positions[i + lag] - x_positions[i]
                    dy = y_positions[i + lag] - y_positions[i]
                    
                    # Calculate squared displacement
                    squared_displacement = dx**2 + dy**2
                    squared_displacements.append(squared_displacement)
        
        # Calculate ensemble average for this lag
        if squared_displacements:
            mean_msd = np.mean(squared_displacements)
            std_error = np.std(squared_displacements) / np.sqrt(len(squared_displacements))
            n_points = len(squared_displacements)
            
            msd_results.append({
                'lag': lag,
                'msd': mean_msd,
                'std_error': std_error,
                'n_points': n_points
            })
    
    return pd.DataFrame(msd_results)

def calculate_flag_percentages(df):
    """Calculate the percentage of each flag type for the Alpha_Flag_Fit column.
    This function groups the DataFrame by 'UID' and counts the occurrences of each flag type.
    """
    uid_flags = df.groupby('UID')['Alpha_Flag_Fit'].first()
    flag_counts = uid_flags.value_counts()
    flag_percentages = (flag_counts / flag_counts.sum() * 100).round(2)
    return flag_percentages


def drop_short_trajectories(df, min_length=const.MIN_TRAJECTORY_LENGTH):
    """
    Drop trajectories that are shorter than a specified minimum length.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns ['Frame', 'X', 'Y', 'UID']
    min_length (int): Minimum number of frames required for a trajectory to be kept
    
    Returns:
    pd.DataFrame: Filtered DataFrame with only trajectories longer than min_length
    """
    # Count the number of frames for each UID
    trajectory_lengths = df.groupby('UID').size()
    
    # Identify UIDs that meet the minimum length requirement
    valid_uids = trajectory_lengths[trajectory_lengths >= min_length].index
    
    # Filter the DataFrame to keep only valid UIDs
    filtered_df = df[df['UID'].isin(valid_uids)].reset_index(drop=True)
    
    return filtered_df

def drop_stationary_trajectories(df, min_displacement=const.LOCALIZATION_PRECISION_NM):
    """
    Drop trajectories that are considered stationary based on their maximum displacement.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns ['Frame', 'X', 'Y', 'UID']
    min_displacement (float): Minimum displacement required for a trajectory to be kept
    
    Returns:
    pd.DataFrame: Filtered DataFrame with only non-stationary trajectories
    """
    # 1. Compute the bounding diameter for each UID
    diameter_per_uid = (
        df.groupby('UID')
        .apply(lambda g: np.sqrt((g['X'].max() - g['X'].min())**2 +
                                (g['Y'].max() - g['Y'].min())**2))
        .rename('diameter')
)
    
    # 2. Find UIDs that actually move beyond the localization precision
    moving_uids = diameter_per_uid[diameter_per_uid >= min_displacement].index

    
    # 3. Filter the original dataframe
    df = df[df['UID'].isin(moving_uids)].copy()
        
    return df

def drop_stationary_trajectories2(df, min_displacement=const.LOCALIZATION_PRECISION_NM):
    """
    Alternative stationary trajectory filtering: Maximum pairwise distance (true diameter)
    If you want the actual maximum distance between any two points in the track. Here 
    Maximum pairwise distance is the largest Euclidean distance between any two points in that trajectory.
    """
    def track_diameter(g):
        coords = g[['X', 'Y']].to_numpy()
        if len(coords) < 2:
            return 0.0
        return pdist(coords).max()
    
    diameter_per_uid = df.groupby('UID').apply(track_diameter).rename('diameter')
    moving_uids = diameter_per_uid[diameter_per_uid >= min_displacement].index
    df = df[df['UID'].isin(moving_uids)].copy()

    return df

def log_binned_histogram(df, column, min_log_d=-4, max_log_d=1, bins_per_decade=10):

    
    num_bins = int((max_log_d - min_log_d) * bins_per_decade)
    np.logspace(min_log_d, max_log_d, num_bins)
    d_vals = df.groupby('UID')[column].first()
    hist, edges = np.histogram(d_vals, bins=np.logspace(min_log_d, max_log_d, num_bins))

    return hist, edges

def linear_binned_histogram(df, column, min_d=0, max_d=2, bin_width=0.01, mean=False):
    """
    Create a linear-binned histogram for a specified column in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing the data.
        column (str): Name of the column to create the histogram for.
        min_d (float): Minimum value for the histogram bins.
        max_d (float): Maximum value for the histogram bins.
        bin_width (float): Width of each bin in the histogram.
        mean (bool): If True, get the mean for each UID instead of the first value.
    
    Returns:
        hist (np.ndarray): Histogram array.
        edges (np.ndarray): Bin edges for the histogram.
    """
    num_bins = int((max_d - min_d) / bin_width)
    edges = np.linspace(min_d, max_d, num_bins + 1)
    if mean:
        d_vals = df.groupby('UID')[column].mean()
    else:
        d_vals = df.groupby('UID')[column].first()
    hist, edges = np.histogram(d_vals, bins=edges)

    return hist, edges

def log_binned_histogram2d(df, x_column, y_column, bins_per_decade=10):
    """
    Create a 2D log-binned histogram for two specified columns in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing the data.
        x_column (str): Name of the column for the x-axis.
        y_column (str): Name of the column for the y-axis.
        x_min_log_d (float): Minimum log value for the x-axis.
        x_max_log_d (float): Maximum log value for the x-axis.
        y_min_log_d (float): Minimum log value for the y-axis.
        y_max_log_d (float): Maximum log value for the y-axis.
        bins_per_decade (int): Number of bins per decade for both axes.
    
    Returns:
        hist (np.ndarray): 2D histogram array.
        x_edges (np.ndarray): Bin edges for the x-axis.
        y_edges (np.ndarray): Bin edges for the y-axis.
    """
    x_vals = df[x_column].to_numpy()
    y_vals = df[y_column].to_numpy()

    x_vals = x_vals[np.isfinite(x_vals) & (x_vals > 0)]
    y_vals = y_vals[np.isfinite(y_vals) & (y_vals > 0)]

    if x_vals.size == 0 or y_vals.size == 0:
        raise ValueError("Both x and y columns must contain at least one positive finite value for log-binned histogram.")

    x_min_log = int(np.floor(np.log10(x_vals.min())))
    x_max_log = int(np.ceil(np.log10(x_vals.max())))
    y_min_log = int(np.floor(np.log10(y_vals.min())))
    y_max_log = int(np.ceil(np.log10(y_vals.max())))

    num_x_bins = max(2, int((x_max_log - x_min_log) * bins_per_decade) + 1)
    num_y_bins = max(2, int((y_max_log - y_min_log) * bins_per_decade) + 1)

    x_edges = np.logspace(x_min_log, x_max_log, num_x_bins)
    y_edges = np.logspace(y_min_log, y_max_log, num_y_bins)
    
    hist, x_edges, y_edges = np.histogram2d(df[x_column], df[y_column], bins=[x_edges, y_edges])
    
    return hist, x_edges, y_edges

def lin_binned_histogram2d(df, x_column, y_column, x_bin_width=0.01, y_bin_width=0.01):
    """
    Create a 2D linear-binned histogram for two specified columns in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing the data.
        x_column (str): Name of the column for the x-axis.
        y_column (str): Name of the column for the y-axis.
        x_bin_width (float): Width of each bin for the x-axis.
        y_bin_width (float): Width of each bin for the y-axis.

    Returns:
        hist (np.ndarray): 2D histogram array.
        x_edges (np.ndarray): Bin edges for the x-axis.
        y_edges (np.ndarray): Bin edges for the y-axis.
    """
    x_vals = df[x_column].to_numpy()
    y_vals = df[y_column].to_numpy()

    x_vals = x_vals[np.isfinite(x_vals) & (x_vals > 0)]
    y_vals = y_vals[np.isfinite(y_vals) & (y_vals > 0)]

    if x_vals.size == 0 or y_vals.size == 0:
        raise ValueError("Both x and y columns must contain at least one positive finite value for linear-binned histogram.")

    x_min = x_vals.min()
    x_max = x_vals.max()
    y_min = y_vals.min()
    y_max = y_vals.max()

    num_x_bins = int((x_max - x_min) / x_bin_width)
    num_y_bins = int((y_max - y_min) / y_bin_width)

    x_edges = np.linspace(x_min, x_max, num_x_bins)
    y_edges = np.linspace(y_min, y_max, num_y_bins)

    hist, x_edges, y_edges = np.histogram2d(df[x_column], df[y_column], bins=[x_edges, y_edges])
    
    return hist, x_edges, y_edges


def domain_diff_type_stacked_df(df):
    """
    Build a stacked-bar-ready DataFrame of diffusion-type proportions per trajectory label.

    This function:
    1. Counts unique trajectories (by ``UID``) for each ``traj_label`` and
       ``Alpha_Flag_Fit`` category.
    2. Removes the ``ignore`` category.
    3. Normalizes counts within each ``traj_label`` to proportions.
    4. Returns a long-format DataFrame with stacked-bar helper columns,
       including label positions and formatted percentage labels.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least the columns:
        ``traj_label``, ``Alpha_Flag_Fit``, and ``UID``.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        - ``traj_label``: trajectory group label.
        - ``Alpha_Flag_Fit``: diffusion-type category.
        - ``Proportion``: normalized proportion in each stack segment.
        - ``bar_bottom``: bottom y-position of each stacked segment.
        - ``label_y_pos``: centered y-position for text labels.
        - ``label``: percentage text (blank for segments < 5%).
    """

    

    # 1. Count trajectories by traj_label and Alpha_Flag_Fit
    counts = df.groupby(['traj_label', 'Alpha_Flag_Fit'])['UID'].nunique().unstack(fill_value=0)
    counts = counts.drop('ignore', axis=1)

    # 2. Normalize each row to get proportions
    normalized = counts.div(counts.sum(axis=1), axis=0)

    # 3. Convert to long format
    plot_df = normalized.reset_index().melt(
        id_vars='traj_label',
        var_name='Alpha_Flag_Fit',
        value_name='Proportion'
    )

    # 4. Compute y-position for labels (middle of each stacked segment)
    plot_df['bar_bottom'] = plot_df.groupby('traj_label')['Proportion'].transform('cumsum') - plot_df['Proportion']
    plot_df['label_y_pos'] = plot_df['bar_bottom'] + plot_df['Proportion'] / 2

    # 5. Format labels as percentages
    plot_df['label'] = (plot_df['Proportion'] * 100).round(1).astype(str) + '%'

    # Optional: hide labels for very thin segments where they won't fit
    plot_df.loc[plot_df['Proportion'] < 0.04, 'label'] = ''

    # Reorder traj_label for plotting
    order = ['in', 'cross', 'out']
    plot_df['traj_label'] = pd.Categorical(plot_df['traj_label'], categories=order, ordered=True)
    plot_df = plot_df.sort_values(['traj_label', 'Alpha_Flag_Fit'])

    return plot_df

def _get_time_averaged_msd(df_trajectory):
    """
    Retrieves the pre-calculated Time-Averaged Mean Squared Displacement (TAMSD)
    for a single particle trajectory from the 'MSD' and 'Lag_T' columns.
    
    Parameters:
        df_trajectory (pd.DataFrame): DataFrame for a single trajectory,
                                      expected to have 'MSD' and 'Lag_T' columns.
                                      
    Returns:
        pd.Series: A Series with 'MSD' values indexed by 'Lag_T'.
                   NaN values in MSD are dropped.
    """
    if 'MSD' not in df_trajectory.columns or 'Lag_T' not in df_trajectory.columns:
        raise ValueError("Input DataFrame for _get_time_averaged_msd must contain 'MSD' and 'Lag_T' columns.")
    
    # The 'MSD' column for a single trajectory is its TAMSD.
    # We set 'Lag_T' as index for easier alignment during ensemble averaging.
    return df_trajectory.set_index('Lag_T')['MSD'].dropna()

def _calculate_ensemble_averaged_msd_average(df_ensemble):
    """
    Calculates the Ensemble-Averaged Mean Squared Displacement (EAMSD)
    by averaging the 'MSD' curves (which are TAMSD for individual trajectories)
    of multiple trajectories in the ensemble.
    
    Parameters:
        df_ensemble (pd.DataFrame): DataFrame containing multiple trajectories,
                                    each with 'UID', 'Lag_T', and 'MSD' columns.
                                    
    Returns:
        pd.Series: EAMSD values indexed by 'Lag_T'. Returns an empty Series if no data.
    """
    if df_ensemble.empty:
        return pd.Series(dtype=float, name='EAMSD_Average')

    # Collect individual MSD curves (TAMSDs) for all unique UIDs in the ensemble
    all_msd_curves = []
    for uid in df_ensemble['UID'].unique():
        # Use the helper to get the TAMSD for each trajectory
        traj_msd_series = _get_time_averaged_msd(df_ensemble[df_ensemble['UID'] == uid])
        if not traj_msd_series.empty:
            all_msd_curves.append(traj_msd_series)

    if not all_msd_curves:
        return pd.Series(dtype=float, name='EAMSD_Average')

    # Concatenate all TAMSD series and calculate the mean for each Lag_T.
    # pd.concat with axis=1 automatically aligns by index (Lag_T).
    combined_msd_series = pd.concat(all_msd_curves, axis=1)
    eamsd_series = combined_msd_series.mean(axis=1)
    eamsd_series.name = 'EAMSD_Average' # Name the series for clarity
    return eamsd_series


def calculate_ergodicity_parameters(df):
    """
    Optimized function to calculate ergodicity parameters for each trajectory based on pre-calculated
    Time-Averaged Mean Squared Displacement (TAMSD) and Ensemble-Averaged
    Mean Squared Displacement (EAMSD).

    The function adds the following columns directly to the input DataFrame:
    - 'EAMSD_All': Ensemble-Averaged MSD across all trajectories for each Lag_T.
    - 'Ergodicity_All': TAMSD / EAMSD_All for each trajectory.
    - 'EAMSD_Group': Ensemble-Averaged MSD for the trajectory's specific group (if 'traj_label' exists).
    - 'Ergodicity_Group': TAMSD / EAMSD_Group for each trajectory within its group.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing single particle tracking data.
                           Must include 'UID', 'Lag_T', and 'MSD' columns.
                           Optionally includes 'traj_label' for group-specific analysis.

    Returns:
        pd.DataFrame: The original DataFrame with added ergodicity-related columns.

    Raises:
        ValueError: If 'MSD' or 'Lag_T' columns are missing.
    """
    if 'MSD' not in df.columns or 'Lag_T' not in df.columns:
        raise ValueError("DataFrame must contain 'MSD' and 'Lag_T' columns for ergodicity calculation.")

    # Calculate global EAMSD once for the entire DataFrame
    global_eamsd_series = _calculate_ensemble_averaged_msd_average(df)
    global_eamsd_dict = global_eamsd_series.to_dict()

    # --- Vectorized calculation for EAMSD_All and Ergodicity_All ---
    # Map 'Lag_T' values to their corresponding global EAMSD values
    df['EAMSD_All'] = df['Lag_T'].map(global_eamsd_dict)
    
    # Calculate Ergodicity_All
    # Ensure division by zero/NaN is handled gracefully; it will result in NaN where EAMSD_All is 0 or NaN
    df['Ergodicity_All'] = df['MSD'] / df['EAMSD_All']

    # --- Conditional calculation for Group EAMSD and Ergodicity_Group ---
    if 'traj_label' in df.columns:
        # Pre-calculate EAMSD for each 'traj_label' group
        group_eamsd_data = []
        for label, group_df in df.groupby('traj_label'):
            group_eamsd_series = _calculate_ensemble_averaged_msd_average(group_df)
            for lag_t, eamsd_val in group_eamsd_series.items():
                group_eamsd_data.append({'traj_label': label, 'Lag_T': lag_t, 'EAMSD_Group_Value': eamsd_val})
        
        if group_eamsd_data:
            group_eamsd_df_lookup = pd.DataFrame(group_eamsd_data)
            
            # Merge this lookup DataFrame back to the original df
            df = df.merge(
                group_eamsd_df_lookup,
                on=['traj_label', 'Lag_T'],
                how='left',
                suffixes=('', '_Group_Merge') # Suffix for the new EAMSD column to avoid collision if any
            )
            df.rename(columns={'EAMSD_Group_Value': 'EAMSD_Group'}, inplace=True)
            
            # Calculate Ergodicity_Group
            df['Ergodicity_Group'] = df['MSD'] / df['EAMSD_Group']
        else:
            # If no group data (e.g., all groups were empty), initialize with NaN
            df['EAMSD_Group'] = np.nan
            df['Ergodicity_Group'] = np.nan
    else:
        # If 'traj_label' doesn't exist, ensure these columns are present but NaN
        df['EAMSD_Group'] = np.nan
        df['Ergodicity_Group'] = np.nan

    return df


def calculate_ergodicity_parameters_old(df):
    """
    Calculates ergodicity parameters for each trajectory based on pre-calculated
    Time-Averaged Mean Squared Displacement (TAMSD) and Ensemble-Averaged
    Mean Squared Displacement (EAMSD).
    
    The function adds the following columns directly to the input DataFrame:
    - 'EAMSD_All': Ensemble-Averaged MSD across all trajectories for each Lag_T.
    - 'Ergodicity_All': TAMSD / EAMSD_All for each trajectory.
    - 'EAMSD_Group': Ensemble-Averaged MSD for the trajectory's specific group (if 'traj_label' exists).
    - 'Ergodicity_Group': TAMSD / EAMSD_Group for each trajectory within its group.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing single particle tracking data.
                           Must include 'UID', 'Lag_T', and 'MSD' columns.
                           Optionally includes 'traj_label' for group-specific analysis.
                           
    Returns:
        pd.DataFrame: The original DataFrame with added ergodicity-related columns.
        
    Raises:
        ValueError: If 'MSD' or 'Lag_T' columns are missing.
    """
    if 'MSD' not in df.columns or 'Lag_T' not in df.columns:
        raise ValueError("DataFrame must contain 'MSD' and 'Lag_T' columns for ergodicity calculation.")

    # Calculate global EAMSD once for the entire DataFrame
    global_eamsd_series = _calculate_ensemble_averaged_msd_average(df)
    global_eamsd_dict = global_eamsd_series.to_dict()

    # Initialize new columns with NaN to ensure they exist before assignment
    df['EAMSD_All'] = np.nan
    df['Ergodicity_All'] = np.nan

    group_eamsd_dicts = {}
    if 'traj_label' in df.columns:
        df['EAMSD_Group'] = np.nan
        df['Ergodicity_Group'] = np.nan
        # Calculate EAMSD for each 'traj_label' group
        for label, group_df in df.groupby('traj_label'):
            group_eamsd_series = _calculate_ensemble_averaged_msd_average(group_df)
            group_eamsd_dicts[label] = group_eamsd_series.to_dict()

    # Apply calculations for each individual trajectory
    for uid, traj_df in df.groupby('UID'):
        # Get the TAMSD for this specific trajectory
        tamsd_series = _get_time_averaged_msd(traj_df)
        
        # If the trajectory's TAMSD is empty, skip to the next UID
        if tamsd_series.empty:
            continue

        # Get the original DataFrame indices for this trajectory that have valid Lag_T values
        # This is crucial for correctly mapping values back to the original df
        valid_lag_t_indices = traj_df[traj_df['Lag_T'].isin(tamsd_series.index)].index

        # Map global EAMSD values to the trajectory's Lag_T values
        eamsd_all_mapped = pd.Series(tamsd_series.index.map(global_eamsd_dict).values, index=tamsd_series.index)
        
        # Calculate Ergodicity_All
        # Handle potential division by zero or NaN if EAMSD_All is zero/NaN
        ergodicity_all_values = tamsd_series / eamsd_all_mapped
        
        # Assign back to the original DataFrame using the original index
        df.loc[valid_lag_t_indices, 'EAMSD_All'] = eamsd_all_mapped.loc[traj_df.loc[valid_lag_t_indices, 'Lag_T']].values
        df.loc[valid_lag_t_indices, 'Ergodicity_All'] = ergodicity_all_values.loc[traj_df.loc[valid_lag_t_indices, 'Lag_T']].values

        if 'traj_label' in df.columns:
            current_group_label = traj_df['traj_label'].iloc[0]
            if current_group_label in group_eamsd_dicts:
                group_eamsd_dict = group_eamsd_dicts[current_group_label]
                eamsd_group_mapped = pd.Series(tamsd_series.index.map(group_eamsd_dict).values, index=tamsd_series.index)

                # Calculate Ergodicity_Group
                ergodicity_group_values = tamsd_series / eamsd_group_mapped

                df.loc[valid_lag_t_indices, 'EAMSD_Group'] = eamsd_group_mapped.loc[traj_df.loc[valid_lag_t_indices, 'Lag_T']].values
                df.loc[valid_lag_t_indices, 'Ergodicity_Group'] = ergodicity_group_values.loc[traj_df.loc[valid_lag_t_indices, 'Lag_T']].values
                
    return df


def calculate_gyration_tensor_parameters(df_trajectory: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates gyration tensor parameters for a single trajectory and adds them to the DataFrame.

    The gyration tensor characterizes the spatial extent and shape of a trajectory.
    Key derived metrics include:
    - Squared Radius of Gyration (Rg^2): A measure of the overall size of the trajectory.
    - Eigenvalues (lambda1, lambda2): Represent the squared lengths of the principal semi-axes
      of the equivalent ellipse that best describes the trajectory's shape.
    - Anisotropy: A dimensionless measure (0 to 1) indicating the deviation from a spherical
      (isotropic, Anisotropy=0) to a linear (highly anisotropic, Anisotropy=1) shape.

    Args:
        df_trajectory (pd.DataFrame): DataFrame containing a single trajectory with 'X' and 'Y' columns.

    Returns:
        pd.DataFrame: The input DataFrame with added columns for gyration tensor parameters:
                      'Radius_of_Gyration_Sq', 'Eigenvalue1', 'Eigenvalue2', 'Anisotropy'.
                      If a trajectory has fewer than 2 points, these columns will be NaN.
    """
    if len(df_trajectory) < 2: # At least 2 points are needed to define a gyration tensor meaningfully
        df_trajectory['Radius_of_Gyration_Sq'] = np.nan
        df_trajectory['Eigenvalue1'] = np.nan
        df_trajectory['Eigenvalue2'] = np.nan
        df_trajectory['Anisotropy'] = np.nan
        return df_trajectory

    # Center the coordinates relative to the trajectory's centroid
    mean_x = df_trajectory['X'].mean()
    mean_y = df_trajectory['Y'].mean()

    dx = df_trajectory['X'] - mean_x
    dy = df_trajectory['Y'] - mean_y

    N = len(df_trajectory)

    # Calculate components of the gyration tensor S_ab = (1/N) * sum(delta_a * delta_b)
    Sxx = np.sum(dx**2) / N
    Syy = np.sum(dy**2) / N
    Sxy = np.sum(dx * dy) / N # Sxy = Syx

    gyration_tensor = np.array([[Sxx, Sxy],
                                [Sxy, Syy]])

    # Calculate eigenvalues of the gyration tensor
    eigenvalues = np.linalg.eigvals(gyration_tensor)
    lambda1 = np.max(eigenvalues) # Larger eigenvalue
    lambda2 = np.min(eigenvalues) # Smaller eigenvalue

    # Calculate Squared Radius of Gyration: Rg^2 = trace(S) = lambda1 + lambda2
    radius_of_gyration_sq = lambda1 + lambda2

    # Calculate Anisotropy: (lambda1 - lambda2) / (lambda1 + lambda2)
    # This metric ranges from 0 (circular/isotropic motion) to 1 (linear/anisotropic motion)
    denominator = (lambda1 + lambda2)
    anisotropy = (lambda1 - lambda2) / denominator if denominator != 0 else np.nan

    # Add the results as new columns to all rows of the current trajectory's DataFrame
    df_trajectory['Radius_of_Gyration_Sq'] = radius_of_gyration_sq
    df_trajectory['Eigenvalue1'] = lambda1
    df_trajectory['Eigenvalue2'] = lambda2
    df_trajectory['Anisotropy'] = anisotropy

    return df_trajectory

def single_mol_fit_single_step(frames, intensities, min_edge=5):
    """
    Fits a single-step photobleaching step-detection model to an intensity
    trace.

    Uses an O(N) cumulative sum algorithm to evaluate all potential step
    partition indices in O(1) time per candidate point. Finds the partition
    index that minimizes the total Residual Sum of Squares (RSS) of a 2-segment
    step model.

    Parameters:
        frames (np.ndarray): 1D array of frame indices corresponding to the trace points.
        intensities (np.ndarray): 1D array of intensity values (e.g., 'mass' or 'signal').
        min_edge (int, optional): Minimum number of consecutive frames required in both 
                                  the 'before' and 'after' plateaus. Defaults to 5.

    Returns:
        dict or None: Returns `None` if the trajectory length is shorter than `2 * min_edge`.
        Otherwise, returns a dictionary with the following fields:
            - 'step_idx' (int): Index in the input array where the step transition occurs.
            - 'bleach_frame' (int/float): Frame value corresponding to the step location.
            - 'mean_before' (float): Mean intensity of the trace before the step.
            - 'mean_after' (float): Mean intensity of the trace after the step.
            - 'step_size' (float): Magnitude of intensity drop (`mean_before - mean_after`).
            - 'noise_before' (float): Standard deviation of intensity before the step.
            - 'noise_after' (float): Standard deviation of intensity after the step.
            - 'rss_step' (float): Total Residual Sum of Squares for the 2-segment step model.
            - 'rss_flat' (float): Total Residual Sum of Squares for a 0-step (flat mean) model.
    """
    N = len(intensities)
    if N < 2 * min_edge:
        return None
    
    cum_sum = np.cumsum(intensities)
    cum_sum_sq = np.cumsum(intensities**2)
    total_sum = cum_sum[-1]
    total_sum_sq = cum_sum_sq[-1]
    
    k_arr = np.arange(min_edge, N - min_edge + 1)
    sum1 = cum_sum[min_edge - 1 : N - min_edge]
    sum2 = total_sum - sum1
    
    mean1 = sum1 / k_arr
    mean2 = sum2 / (N - k_arr)
    
    rss1 = cum_sum_sq[min_edge - 1 : N - min_edge] - (sum1**2) / k_arr
    rss2 = (total_sum_sq - cum_sum_sq[min_edge - 1 : N - min_edge]) - (sum2**2) / (N - k_arr)
    total_rss = rss1 + rss2
    
    best_idx = np.argmin(total_rss)
    best_k = k_arr[best_idx]
    mean_before = mean1[best_idx]
    mean_after = mean2[best_idx]
    
    y_before = intensities[:best_k]
    y_after = intensities[best_k:]
    
    return {
        'step_idx': best_k,
        'bleach_frame': frames[best_k],
        'mean_before': mean_before,
        'mean_after': mean_after,
        'step_size': mean_before - mean_after,
        'noise_before': np.std(y_before) if len(y_before) > 1 else 1e-6,
        'noise_after': np.std(y_after) if len(y_after) > 1 else 1e-6,
        'rss_step': total_rss[best_idx],
        'rss_flat': total_sum_sq - (total_sum**2) / N
    }


def single_mol_analyze_single_step_photobleaching(
    df,
    intensity_col='mass',
    min_track_length=15,
    min_edge_frames=5,
    min_snr=3.0,
    max_final_ratio=0.4
):
    """
    Runs single-step photobleaching analysis on all trajectories and fits statistical models.

    Iterates over all trajectories in the dataset, applying single-step partition fitting
    `single_mol_fit_single_step` and enforcing single-molecule photobleaching selection rules:
    1. Positive intensity drop (`step_size > 0`).
    2. High Signal-to-Noise Ratio (SNR ≥ `min_snr`).
    3. Bleaching down to background level (`mean_after` < `mean_before` × `max_final_ratio`).

    Subsequent statistical model fitting is performed on qualifying single-step events:
    - **Step Sizes (ΔI)**: Fitted with a Gaussian function to extract quantum unit loss.
    - **Bleaching Lifetime (τ)**: Fitted with a single-exponential decay function.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing trajectory data. Must include columns
                           'UID', 'Frame', and the specified `intensity_col`.
        intensity_col (str, optional): Intensity metric column name (e.g., 'mass', 'raw_mass',
                                       or 'signal'). Defaults to 'mass'.
        min_track_length (int, optional): Minimum number of frames required to analyze a trajectory.
                                          Defaults to 15.
        min_edge_frames (int, optional): Minimum edge frames required before and after the step.
                                         Defaults to 5.
        min_snr (float, optional): Minimum Signal-to-Noise Ratio (`step_size / noise_after`).
                                  Defaults to 3.0.
        max_final_ratio (float, optional): Maximum allowable ratio of mean intensity after the step
                                           to mean intensity before the step. Defaults to 0.4.

    Returns:
        tuple[pd.DataFrame, dict]:
            - **bleach_df** (pd.DataFrame): DataFrame of qualifying photobleaching trajectories
              containing step parameters ('UID', 'step_idx', 'bleach_frame', 'mean_before',
              'mean_after', 'step_size', 'noise_before', 'noise_after', 'snr', 'track_length',
              'bleach_time').
            - **fit_summary** (dict): Dictionary containing statistical fit outputs:
                - `'gauss'`: Tuple of `(counts, bin_edges, popt_gauss)` for step size distribution.
                - `'exp'`: Tuple of `(counts_t, bin_edges_t, popt_exp)` for survival time decay distribution.

    """
    bleach_results = []
    all_tracks = df.groupby('UID')
    
    for uid, track_df in all_tracks:
        track_df = track_df.sort_values('Frame').dropna(subset=[intensity_col])
        if len(track_df) < min_track_length:
            continue
        
        frames = track_df['Frame'].values
        intensities = track_df[intensity_col].values
        
        fit = single_mol_fit_single_step(frames, intensities, min_edge=min_edge_frames)
        if fit is None:
            continue
            
        snr = fit['step_size'] / fit['noise_after']
        is_bleached = (fit['mean_after'] < fit['mean_before'] * max_final_ratio)
        
        if fit['step_size'] > 0 and snr >= min_snr and is_bleached:
            fit['UID'] = uid
            fit['snr'] = snr
            fit['track_length'] = len(track_df)
            fit['bleach_time'] = fit['step_idx'] * const.DT
            bleach_results.append(fit)

    bleach_df = pd.DataFrame(bleach_results)
    fit_summary = {'gauss': None, 'exp': None}
    
    if bleach_df.empty:
        print("Warning: No single-step photobleaching tracks found.")
        return bleach_df, fit_summary

    # Summary prints
    n_analyzed = sum(1 for _, t in all_tracks if len(t) >= min_track_length)
    print("=" * 60)
    print("SINGLE-STEP PHOTOBLEACHING ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Analyzed Trajectories (>= {min_track_length} frames): {n_analyzed}")
    print(f"Single-step Bleach Events:           {len(bleach_df)} ({len(bleach_df)/n_analyzed*100:.1f}%)")
    print(f"Average Step Size (ΔI):              {bleach_df['step_size'].mean():.2f} ± {bleach_df['step_size'].std():.2f}")
    print(f"Average SNR:                         {bleach_df['snr'].mean():.2f}")
    print("=" * 60)

    # Gaussian Fit on Step Sizes
    step_sizes = bleach_df['step_size'].values
    counts, bin_edges = np.histogram(step_sizes, bins='auto')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    try:
        popt_gauss, _ = curve_fit(
            lambda x, a, x0, s: a * np.exp(-(x - x0)**2 / (2 * s**2)),
            bin_centers, counts,
            p0=[max(counts), np.median(step_sizes), np.std(step_sizes)]
        )
        fit_summary['gauss'] = (counts, bin_edges, popt_gauss)
    except (RuntimeError, ValueError) as e:
        print(f"Gaussian fit failed: {e}")
        fit_summary['gauss'] = (counts, bin_edges, None)

    # Exponential Decay Fit on Bleaching Times
    bleach_times = bleach_df['bleach_time'].values
    counts_t, bin_edges_t = np.histogram(bleach_times, bins='auto')
    bin_centers_t = (bin_edges_t[:-1] + bin_edges_t[1:]) / 2
    try:
        popt_exp, _ = curve_fit(
            lambda t, a, tau: a * np.exp(-t / tau),
            bin_centers_t, counts_t,
            p0=[max(counts_t), np.mean(bleach_times)]
        )
        fit_summary['exp'] = (counts_t, bin_edges_t, popt_exp)
    except (RuntimeError, ValueError) as e:
        print(f"Exponential decay fit failed: {e}")
        fit_summary['exp'] = (counts_t, bin_edges_t, None)

    return bleach_df, fit_summary

def single_mol_fit_multi_step(frames, intensities, min_edge=5, max_steps=3, bic_threshold=10):
    """
    Fits a multi-step photobleaching model to an intensity trace using
    recursive CUSUM partition search with BIC model selection.

    Starting from a flat (0-step) model, the algorithm iteratively searches
    for the best additional step within each existing segment by calling
    `single_mol_fit_single_step`. At each iteration the candidate step that
    yields the largest RSS reduction is accepted only if the Bayesian
    Information Criterion (BIC) improvement exceeds `bic_threshold`:

        BIC = N · ln(RSS / N) + p · ln(N)

    where N is the trace length and p is the number of free parameters
    (p = 2k + 1 for k steps: k + 1 plateau means plus k step locations).
    Iteration stops when no candidate improves BIC by more than the
    threshold or `max_steps` is reached.

    Parameters:
        frames (np.ndarray): 1D array of frame indices for the trace.
        intensities (np.ndarray): 1D array of intensity values.
        min_edge (int, optional): Minimum frames required in each plateau
                                  segment. Defaults to 5.
        max_steps (int, optional): Maximum number of steps to fit.
                                   Defaults to 3.
        bic_threshold (float, optional): Minimum ΔBIC required to accept an
                                         additional step. Defaults to 10.

    Returns:
        dict or None: Returns `None` if the trace is shorter than
        `2 * min_edge`. Otherwise returns a dictionary with:
            - 'num_steps' (int): Number of accepted steps (0 if flat model wins).
            - 'step_locations_idx' (list[int]): Sorted array indices of step
              transitions.
            - 'step_frames' (list[int/float]): Frame values at each step.
            - 'plateau_values' (list[float]): Mean intensity of each plateau
              segment (length = num_steps + 1).
            - 'step_sizes' (list[float]): Intensity drop at each step
              (plateau_i − plateau_{i+1}, positive for downward steps).
    """
    N = len(intensities)
    if N < 2 * min_edge:
        return None
    
    best_rss = np.sum((intensities - np.mean(intensities))**2)
    best_p = 1
    best_bic = N * np.log(best_rss / N) + best_p * np.log(N)
    
    step_locations = []
    
    for step_num in range(1, max_steps + 1):
        candidate_fits = []
        boundaries = [0] + sorted(step_locations) + [N]
        
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            seg_frames = frames[start:end]
            seg_intensities = intensities[start:end]
            
            fit = single_mol_fit_single_step(seg_frames, seg_intensities, min_edge=min_edge)
            if fit is not None:
                other_segments_rss = sum(
                    np.sum((intensities[boundaries[j]:boundaries[j+1]] - np.mean(intensities[boundaries[j]:boundaries[j+1]]))**2)
                    for j in range(len(boundaries) - 1) if j != i
                )
                total_candidate_rss = other_segments_rss + fit['rss_step']
                candidate_fits.append((total_candidate_rss, start + fit['step_idx']))
        
        if not candidate_fits:
            break
            
        candidate_fits.sort()
        best_candidate_rss, best_candidate_loc = candidate_fits[0]
        
        candidate_p = best_p + 2
        candidate_bic = N * np.log(best_candidate_rss / N) + candidate_p * np.log(N)
        
        if (best_bic - candidate_bic) > bic_threshold:
            step_locations.append(best_candidate_loc)
            best_rss = best_candidate_rss
            best_bic = candidate_bic
            best_p = candidate_p
        else:
            break
            
    boundaries_sorted = sorted(step_locations)
    boundaries = [0] + boundaries_sorted + [N]
    final_plateaus = [np.mean(intensities[boundaries[i]:boundaries[i+1]]) for i in range(len(boundaries)-1)]
    
    return {
        'num_steps': len(step_locations),
        'step_locations_idx': boundaries_sorted,
        'step_frames': [frames[loc] for loc in boundaries_sorted],
        'plateau_values': final_plateaus,
        'step_sizes': [-np.diff(final_plateaus)[i] for i in range(len(final_plateaus)-1)]
    }


def single_mol_analyze_multi_step_photobleaching(
    df,
    intensity_col='mass',
    min_track_length=15,
    min_edge_frames=5,
    max_steps=5,
    bic_threshold=10.0,
    max_final_ratio=0.4
):
    """
    Runs recursive multi-step photobleaching analysis on all trajectories
    and evaluates stoichiometry.

    Iterates over every trajectory in the dataset, applying
    `single_mol_fit_multi_step` (recursive CUSUM + BIC model selection)
    and enforcing the following quality filters:
    1. All detected steps are downward (every ΔI > 0).
    2. Final plateau is near background
       (Ī_last / Ī_first < `max_final_ratio`).

    Qualifying trajectories are classified by step count into
    stoichiometry classes (1-step → monomer, 2-step → dimer,
    ≥ 3-step → trimer / aggregate) and a summary is printed.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing trajectory data.
                           Must include columns 'UID', 'Frame', and the
                           specified `intensity_col`.
        intensity_col (str, optional): Intensity metric column name
                                       (e.g., 'mass', 'raw_mass', or
                                       'signal'). Defaults to 'mass'.
        min_track_length (int, optional): Minimum number of frames
                                          required to analyze a trajectory.
                                          Defaults to 15.
        min_edge_frames (int, optional): Minimum frames required in each
                                         plateau segment passed to the
                                         step fitter. Defaults to 5.
        max_steps (int, optional): Maximum number of photobleaching steps
                                   to search for per trajectory.
                                   Defaults to 5.
        bic_threshold (float, optional): Minimum ΔBIC required to accept
                                         an additional step. Defaults to
                                         10.0.
        max_final_ratio (float, optional): Maximum allowable ratio
                                           Ī_last / Ī_first for a
                                           trajectory to qualify as fully
                                           bleached. Defaults to 0.4.

    Returns:
        pd.DataFrame: DataFrame of qualifying multi-step photobleaching
        trajectories with columns:
            - 'UID': Trajectory identifier.
            - 'num_steps' (int): Number of detected steps.
            - 'step_locations_idx' (list[int]): Array indices of step
              transitions.
            - 'step_frames' (list[int/float]): Frame values at each step.
            - 'plateau_values' (list[float]): Mean intensity of each
              plateau segment.
            - 'step_sizes' (list[float]): Intensity drop at each step.
            - 'track_length' (int): Number of frames in the trajectory.
            - 'final_ratio' (float): Ī_last / Ī_first intensity ratio.

        Returns an empty DataFrame if no trajectories pass the filters.
    """
    step_results = []
    all_tracks = df.groupby('UID')

    for uid, track_df in all_tracks:
        track_df = track_df.sort_values('Frame').dropna(subset=[intensity_col])
        if len(track_df) < min_track_length:
            continue
        
        frames = track_df['Frame'].values
        intensities = track_df[intensity_col].values
        
        fit = single_mol_fit_multi_step(
            frames, intensities, 
            min_edge=min_edge_frames, 
            max_steps=max_steps, 
            bic_threshold=bic_threshold
        )
        if fit is None or fit['num_steps'] == 0:
            continue
            
        all_downward = all(size > 0 for size in fit['step_sizes'])
        final_ratio = fit['plateau_values'][-1] / fit['plateau_values'][0] if fit['plateau_values'][0] != 0 else np.inf
        is_bleached = final_ratio < max_final_ratio
        
        if all_downward and is_bleached:
            fit['UID'] = uid
            fit['track_length'] = len(track_df)
            fit['final_ratio'] = final_ratio
            step_results.append(fit)

    results_df = pd.DataFrame(step_results)

    if results_df.empty:
        print("Warning: No stepwise bleaching trajectories qualified under specified rules.")
        return results_df

    n_analyzed_tracks = sum(1 for _, t in all_tracks if len(t) >= min_track_length)
    n_stepwise = len(results_df)
    
    print("\n" + "="*60)
    print("RECURSIVE MULTI-STEP PHOTOBLEACHING ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total trajectories in dataset:           {len(all_tracks)}")
    print(f"Trajectories >= {min_track_length} frames analyzed:      {n_analyzed_tracks}")
    print(f"Qualifying stepwise bleaching tracks:    {n_stepwise} ({100 * n_stepwise / n_analyzed_tracks:.1f}% of analyzed)")
    
    step_counts = results_df['num_steps'].value_counts().sort_index()
    for steps, count in step_counts.items():
        pct = 100 * count / n_stepwise
        label = "Monomer" if steps == 1 else "Dimer" if steps == 2 else "Trimer/Aggregate"
        print(f"  {steps}-step class ({label}): {count} tracks ({pct:.1f}%)")
    print("="*60 + "\n")

    return results_df