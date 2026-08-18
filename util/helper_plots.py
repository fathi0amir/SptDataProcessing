import holoviews as hv
import numpy as np

import util.look_and_feel as laf


def temporal_intensity_histograms(df, traj1=1, traj2=9, traj3=30, traj4=80, hv_bokeh_theme=None):

    intensity_col = "mass"

    # Ensure trajectories are sorted by frame for each UID
    df_sorted = df.sort_values(["UID", "Frame"]) if "Frame" in df.columns else df

    # Extract the 1st, 10th, and 20th element's mass for each UID
    traj_1_int = df_sorted.groupby("UID")[intensity_col].nth(traj1).dropna()
    traj_2_int = df_sorted.groupby("UID")[intensity_col].nth(traj2).dropna()
    traj_3_int = df_sorted.groupby("UID")[intensity_col].nth(traj3).dropna()
    traj_4_int = df_sorted.groupby("UID")[intensity_col].nth(traj4).dropna()

    # Set Bokeh theme
    hv.renderer("bokeh").theme = hv_bokeh_theme if hv_bokeh_theme else "default"

    # Function to generate HoloViews histogram with xlim (0, 800)
    def create_mass_hist(data, title, color):
        counts, bin_edges = np.histogram(data, bins="auto", range=(0, 800))
        hist = (
            hv.Histogram(
                (counts, bin_edges), kdims=["Mass (a.u.)"], vdims=["Frequency"]
            )
            .opts(
                title=title,
                color=color,
                line_color="#FFFFFF99",
                alpha=0.6,
                xlim=(0, 500),
                width=380,
                height=280,
            )
            .opts(laf.HV_BOKEH_BASIC)
            .opts(hooks=[laf.bokeh_add_topright_linear_axes])
            .opts(tools=["hover"], active_tools=[])
        )
        return hist

    # Create 4 separate plots
    plot_1 = create_mass_hist(
        traj_1_int, f"Intensities for Traj. {traj1} (N={len(traj_1_int)})", "#3498DB"
    )
    plot_2 = create_mass_hist(
        traj_2_int, f"Intensities for Traj. {traj2} (N={len(traj_2_int)})", "#E67E22"
    )
    plot_3 = create_mass_hist(
        traj_3_int, f"Intensities for Traj. {traj3} (N={len(traj_3_int)})", "#2ECC71"
    )
    plot_4 = create_mass_hist(
        traj_4_int, f"Intensities for Traj. {traj4} (N={len(traj_4_int)})", "#9B59B6"
    )

    # Combine into HoloViews Layout with 4 plots
    mass_dashboard = (
        (plot_1 + plot_2 + plot_3 + plot_4)
        .opts(
            shared_axes=False,
            axiswise=True,
            framewise=True,
            tabs=True,
        )
    )

    return mass_dashboard