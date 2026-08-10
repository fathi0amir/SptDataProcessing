# Single Particle Tracking (SPT) Analysis Notes

## Single-Step Photobleaching Analysis

### 1. Scientific Principles & Significance
To prove that individual fluorescent spots in Single Particle Tracking (SPT)
represent **single molecules** (monomers, e.g., Cy5-DOPE) rather than
multi-molecule aggregates, **single-step photobleaching** is the most rigorous
and widely accepted standard in literature.

* **Intensity Quantization (Unimodality):** A single dye molecule acts as a
  quantum emitter—it is either "on" (emitting photons) or "off" (permanently
  bleached). When it photobleaches, it goes from a stable intensity state
  straight to the background level in a single, discrete step. 
  * If a spot contains a single fluorophore, it will bleach in exactly **1
    step**.
  * If a spot contains $N$ co-localized fluorophores within the diffraction
    limit, it will bleach in $N$ discrete steps.
  * A histogram of bleaching step sizes ($\Delta I$) across a large population
    of molecules should yield a narrow, unimodal distribution representing the
    **fluorescence quantum** of a single Cy5 emitter.
* **Single-Exponential Survival Decay:** Photobleaching is a stochastic
  first-order reaction. For a pure population of single emitters, the
  probability of remaining unbleached over time decays exponentially: $$N(t) =
  N_0 e^{-t/\tau}$$ where $\tau$ is the characteristic photobleaching lifetime.
  Multi-fluorophore spots would deviate from this single-exponential decay
  (showing multi-exponential behavior or a lag phase).

---

### 2. Step Detection & Selection Algorithm

The analysis uses an $O(N)$ Cumulative Sum (CUSUM) Least-Squares Partitioning
algorithm to split each trajectory into two states (Before and After bleaching)
and evaluates whether it meets single-molecule criteria.

#### Step Partitioning Algorithm:
For a trajectory of intensity measurements $y_1, y_2, \dots, y_N$:
1. For every candidate split index $k \in [\text{min\_edge}, N -
   \text{min\_edge}]$, calculate the mean before ($\mu_1$) and after ($\mu_2$)
   the split.
2. Calculate the Residual Sum of Squares (RSS): $$\text{RSS}(k) = \sum_{i=1}^{k}
   (y_i - \mu_1)^2 + \sum_{i=k+1}^{N} (y_i - \mu_2)^2$$
3. The optimal bleach frame is the index $k$ that minimizes $\text{RSS}(k)$.

#### Vectorized $O(N)$ Implementation Details:
To process thousands of trajectories in seconds, the algorithm avoids slow Python loops and computes the RSS for all candidate split points simultaneously using vectorized NumPy operations:

* **Cumulative Sums (CUSUM):** Precomputes the running sum of intensities and squared intensities in $O(N)$ time:
  $$\text{cum\_sum}[j] = \sum_{i=1}^{j} y_i, \quad \text{cum\_sum\_sq}[j] = \sum_{i=1}^{j} y_i^2$$
  This allows the sum, mean, and sum of squares of *any* slice of the trajectory to be calculated in $O(1)$ constant time.
* **Vectorized Means:** Calculates the mean before ($\mu_1$) and after ($\mu_2$) for all candidate split points $k$ at once:
  $$\mu_1[k] = \frac{\text{cum\_sum}[k]}{k}, \quad \mu_2[k] = \frac{\text{total\_sum} - \text{cum\_sum}[k]}{N - k}$$
* **Vectorized RSS Calculation:** Uses the algebraic identity $\sum (y_i - \mu)^2 = \sum y_i^2 - \frac{(\sum y_i)^2}{n}$ to compute the RSS for both segments in a single vectorized step:
  $$\text{RSS}_1[k] = \text{cum\_sum\_sq}[k] - \frac{(\text{cum\_sum}[k])^2}{k}$$
  $$\text{RSS}_2[k] = (\text{total\_sum\_sq} - \text{cum\_sum\_sq}[k]) - \frac{(\text{total\_sum} - \text{cum\_sum}[k])^2}{N - k}$$
  $$\text{RSS}_{\text{total}}[k] = \text{RSS}_1[k] + \text{RSS}_2[k]$$
* **Optimal Split Selection:** Uses `np.argmin(RSS_total)` to find the exact frame $k$ where the RSS is minimized, representing the mathematically most likely photobleaching event.
* **Noise & SNR Estimation:** Calculates the standard deviation of the intensity before and after the step. The post-step standard deviation ($\sigma_{\text{after}}$) represents the background camera noise, which is used to calculate the Signal-to-Noise Ratio ($\text{SNR} = \Delta I / \sigma_{\text{after}}$).

#### Single-Molecule Selection Rules (Filters):
To ensure high-confidence single bleach events and filter out noisy or
multi-step tracks, trajectories must satisfy:
1. **Minimum Length (`MIN_TRACK_LENGTH = 15`):** Trajectories must have enough
   frames to establish reliable pre- and post-bleach baselines.
2. **Downward Step (`step_size > 0`):** The average intensity before the step
   must be higher than after ($\mu_1 > \mu_2$).
3. **Signal-to-Noise Ratio (`MIN_SNR = 3.0`):** The intensity drop must be sharp
   compared to the background noise level ($\sigma_{\text{after}}$):
   $$\text{SNR} = \frac{\Delta I}{\sigma_{\text{after}}} \ge 3.0$$
4. **Complete Bleaching (`MAX_FINAL_RATIO = 0.4`):** The intensity after the
   step must drop to the background level: $$\frac{\mu_2}{\mu_1} < 0.4$$

---

### 3. Interpreting the Dashboard Plots

The dashboard displays six subplots to provide a complete picture of the
photobleaching characteristics:

* **Subplots 1–4: Representative Single-Step Bleaching Tracks**
  * **Blue markers and lines:** Raw intensity values over time.
  * **Red step-line:** Overlaid model fit. A clean step profile shows a flat
    "on" state followed by an instantaneous drop to a flat background level.
  * **Annotations:** Detail the unique trajectory ID (`UID`), calculated step
    size ($\Delta I$), and Signal-to-Noise Ratio (`SNR`).
* **Subplot 5: Bleaching Step Size Distribution**
  * Shows a histogram of the intensity drops ($\Delta I$) across all qualifying
    tracks.
  * Overlaid with a **Gaussian Fit** (green curve) to find the mean step size
    $\mu$ and standard deviation $\sigma$. A narrow, unimodal Gaussian centered
    around a clean intensity value validates that you are observing a uniform
    population of single dye emitters.
* **Subplot 6: Photobleaching Lifetime Decay**
  * Shows a histogram of the survival times (the time elapsed before the
    bleaching event occurs).
  * Overlaid with a **Single-Exponential Fit** (purple curve) to determine the
    characteristic photobleaching lifetime ($\tau$). A clean exponential decay
    profile is strong statistical proof of a uniform single-molecule population.

---

### 4. Adjusting Parameters for Different Datasets
If your background noise changes or you use different lasers/exposures, adjust
these parameters in the code cell:
* **`INTENSITY_COL`**: Set to `'mass'`, `'raw_mass'`, or `'signal'` depending on
  which metric has the highest SNR.
* **`MIN_SNR`**: Lower to `2.5` if you have low excitation power; raise to `4.0`
  or `5.0` for ultra-clean publication-ready step selection.
* **`MAX_FINAL_RATIO`**: If background autofluorescence is high, you might need
  to increase this slightly (e.g., to `0.5`).

---

### 5. Key Literature References
To support your single-molecule claims in peer-reviewed journals (such as *Biophysical Journal*, *Nature Methods*, or *PNAS*), you can cite these foundational and highly respected publications that establish single-step photobleaching as the gold standard:

1. **Foundational Single-Step Photobleaching Proof:**
   * **Reference:** Funatsu, T., Harada, Y., Tokunaga, M., Saito, K., & Yanagida, T. (1995). *Imaging of single fluorescent molecules and individual ATP turnovers by single myosin molecules in aqueous solution.* **Nature**, 374(6522), 555-559.
   * **Significance:** One of the earliest and most famous papers demonstrating that single fluorophores (like Cy3/Cy5) bleach in a single, quantized step to the background, establishing the physical basis of single-molecule imaging.

2. **Quantifying Stoichiometry and Aggregation in Membranes:**
   * **Reference:** Ulbrich, M. H., & Isacoff, E. Y. (2007). *Subunit counting in membrane-bound proteins.* **Nature Methods**, 4(4), 319-321.
   * **Significance:** Establishes the exact statistical framework for using step-wise photobleaching to count subunits and prove the absence of aggregation or oligomerization in membrane-bound systems.

3. **Standard SPT Methodology and Validation:**
   * **Reference:** Manzo, C., & Garcia-Parajo, M. F. (2015). *A review of progress in single particle tracking: from methods to biophysical insights.* **Reports on Progress in Physics**, 78(12), 124601.
   * **Significance:** A comprehensive review detailing standard validation protocols in SPT, highlighting single-step photobleaching as the primary method to confirm single-molecule tracking.

4. **Step Detection Algorithms in Single-Molecule Biophysics:**
   * **Reference:** Carter, B. C., Vershinin, M., & Gross, S. P. (2008). *A Comparison of Step-Detection Methods: How Well Can You Do?* **Biophysical Journal**, 94(1), 306-319.
   * **DOI:** [10.1529/biophysj.107.110601](https://doi.org/10.1529/biophysj.107.110601)
   * **Significance:** This is the definitive paper systematically comparing step-detection algorithms (including the chi-squared/least-squares partitioning method used in your code) for single-molecule trajectories. It validates the statistical rigor of the change-point detection approach for identifying discrete steps in noisy data.
