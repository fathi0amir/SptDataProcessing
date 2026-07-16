"""
This file contains all the constants used in the project.
"""

DT = 0.02  # Time interval between frames in seconds
MSD_LENGTH_DIVISOR = 0.5  # The maximum lag time is set to 60% of the total time
# Thresholds for alpha values
PIXEL_IN_NM = 65  # Conversion factor from pixels to nanometers


# Nanometer to micrometer conversion factor if needed
# If the trackmate data is in nanometers put this value to 0.001
# otherwise, if the data is in micrometers, set it to 1
NANOMETER_TO_MICROMETER = 0.001  # Conversion factor from nanometers to micrometers

LOCALIZATION_PRECISION_NM = 20 # Localization precision in nanometers, used for filtering stationary tracks.
LOCALIZATION_PRECISION_UM = LOCALIZATION_PRECISION_NM * NANOMETER_TO_MICROMETER  # Localization precision in micrometers
MIN_TRAJECTORY_LENGTH = 40 # Minimum number of frames for a trajectory to be considered valid

ALPHA_THRESHOLDS = {
    # between 0.8 and 1.2 is considered normal diffusion
    'ignore': 0.3,  # Ignore tracks with alpha <= 0.3
    'sub': 0.8,     # Low alpha threshold
    'sup': 1.2,     # High alpha threshold
}
MSD_SLOPE_POINTS = 4  # Number of points to consider for the slope in MSD plot to classify alpha
MSD_PLOT_POINTS = 6 # Number of points to plot in the MSD plot

# Constants for the confinement analysis
# log(ψ) = 0.2048 - 2.5117 ⨉ D ⨉ Δt / R²
# where ψ is the confinement probability, D is the diffusion coefficient, 
# Δt is the time interval, and R is the radius of confinement
PROBABILITY_THRESHOLD = 0.9 # Less than 10% particle is in nonrandom motion
WINDOW_SIZE = 40
CONFINEMENT_TIME_THRESHOLD = 0.1
CONFINEMENT_PROBABILITY_THRESHOLD = 0.5
MSD_FIT_POINTS = 4  # Number of points to fit the MSD curve for confinement analysis
CONF_C1 = 0.2048  # Constant for the confinement analysis equation
CONF_C2 = 2.5117  # Constant for the confinement analysis equation

TMSD_WINDOW_SIZE = 40  # Window size for transient MSD calculation
TMSD_FIT_POINTS = 4  # Number of points to fit the TMSD curve

# I find this through trial and error. This divided by the number of 
# trajectories gives the opacity value for the plot
OPACITY_PARAM = 16 