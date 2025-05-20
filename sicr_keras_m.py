#!/usr/bin/env python3
"""
Signal to Interference Cancellation Ratio (SICR) Model
Keras Model Implementation
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging

import numpy as np
import tensorflow as tf
import sionna as sn
import matplotlib.pyplot as plt

from tdl import TDL
from sionna.phy import PI
from sionna.phy.utils import db_to_lin, ebnodb2no
from sionna.phy.ofdm.pilot_pattern import PilotPattern
from sionna.phy.channel import ApplyOFDMChannel, AWGN
from sionna.phy.channel.utils import cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.ofdm.resource_grid import (
    ResourceGrid,
    ResourceGridMapper,
    ResourceGridDemapper,
    RemoveNulledSubcarriers,
)
from sionna.phy.ofdm.channel_estimation import LSChannelEstimator
from sionna.phy.signal.window import CustomWindow
from sionna.phy.signal.utils import fft, ifft
from sionna.phy.mapping import Mapper, Demapper, BinarySource, Constellation
from sionna.phy.utils.metrics import compute_ber


class SICRKerasModel(tf.keras.Model):
    """
    Signal to Interference Cancellation Ratio (SICR) Keras Model

    This class implements a model for calculating and analyzing the Signal to
    Interference Cancellation Ratio in wireless communication systems using Sionna
    and Keras Model class for better organization.
    """

    def __init__(self, config=None, **kwargs):
        """
        Initialize the SICR model with the given parameters.

        Args:
            config: Dictionary with configuration parameters
            **kwargs: Additional keyword arguments to override config values
        """
        super().__init__()

        # Default configuration parameters
        self.config = {
            "seed": 42,
            "batch_size": 1,
            "num_tx": 1,
            "num_streams_per_tx": 1,
            "num_ofdm_symbols": 5,
            "num_bits_per_symbol": 10,
            "fft_size": 8192,
            "num_useful_subcarriers": 6913,
            "subcarrier_spacing": 1220.703125,
            "cyclic_prefix_length": 0,
            "delay_spread": 1.0e-6,
            "carrier_frequency": 6e5,
            "loopback_amplification": 30,
            "iter": 7,
            "iter_n": 8,
            "iter_n2": 8,
            "delay": 11e-6,  # Added delay parameter for convenience
        }

        # Update configuration with provided config and kwargs
        if config is not None:
            self.config.update(config)

        # Override with any kwargs provided
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value

        # Set random seed
        sn.phy.config.seed = self.config["seed"]

        # Calculate guard carriers
        self._calculate_guard_carriers()

        # Initialize system components
        self._initialize_components()

    def _calculate_guard_carriers(self):
        """Calculate guard carriers based on FFT size and useful subcarriers"""
        self.config["num_guard_carrier"] = (
            self.config["fft_size"] - self.config["num_useful_subcarriers"]
        )
        self.config["guard_carrier_left"] = int(
            (self.config["num_guard_carrier"] - 1) / 2
        )
        self.config["guard_carrier_right"] = int(
            (self.config["num_guard_carrier"] - 1) / 2
        )

    def _initialize_components(self):
        """Initialize all system components as Keras layers or custom objects"""
        # Generate pilot pattern
        self.pilot_pattern = self._generate_pilot_pattern()

        # Create resource grid
        self.resource_grid = self._create_resource_grid()

        # Create binary source, constellation and mappers
        self.binary_source = BinarySource()
        self.constellation = Constellation(
            constellation_type="qam",
            num_bits_per_symbol=self.config["num_bits_per_symbol"],
            normalize=True,
        )

        # Create signal processing components
        self._create_signal_processors()

        # Create channel models
        self._create_channel_models()

    def _generate_pilot_pattern(self):
        """Generate pilot pattern based on configuration"""
        # Standard PRBS pattern (truncated for readability)
        prbs_ptt = [
            1,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
        ]

        # Configuration extraction
        num_tx = self.config["num_tx"]
        num_streams_per_tx = self.config["num_streams_per_tx"]
        nSymb = self.config["num_ofdm_symbols"]
        nCarrier = self.config["num_useful_subcarriers"]
        A_sp_dB = 3  # Default pilot amplitude in dB
        d_x = 16  # Default frequency spacing

        # Convert dB value to linear scale
        A_sp = np.sqrt(10 ** (A_sp_dB / 10))

        # Define mask and pilot arrays
        mask = np.zeros([num_tx, num_streams_per_tx, nSymb, nCarrier], dtype=bool)
        mask[..., 0:nCarrier:d_x] = True

        prbs_ext = tf.tile(tf.constant(prbs_ptt, dtype=tf.float32), [nSymb])

        pilots = np.zeros(
            [num_tx, num_streams_per_tx, len(prbs_ext)], dtype=np.complex64
        )
        real_part = 2 * A_sp * (0.5 - prbs_ext[: len(pilots[0, 0])])
        pilots[..., :] = tf.complex(real_part, tf.zeros_like(real_part))

        return PilotPattern(mask=mask, pilots=pilots)

    def _create_resource_grid(self):
        """Create the resource grid using the configuration"""
        return ResourceGrid(
            num_ofdm_symbols=self.config["num_ofdm_symbols"],
            fft_size=self.config["fft_size"],
            subcarrier_spacing=self.config["subcarrier_spacing"],
            num_tx=self.config["num_tx"],
            num_streams_per_tx=self.config["num_streams_per_tx"],
            cyclic_prefix_length=self.config["cyclic_prefix_length"],
            num_guard_carriers=(
                self.config["guard_carrier_left"],
                self.config["guard_carrier_right"],
            ),
            dc_null=True,
            pilot_pattern=self.pilot_pattern,
        )

    def _create_signal_processors(self):
        """Create signal processing components"""
        # Mappers
        self.mapper = Mapper(
            constellation_type="custom",
            num_bits_per_symbol=self.config["num_bits_per_symbol"],
            constellation=self.constellation,
        )

        self.demapper = Demapper(
            demapping_method="app",
            constellation_type="custom",
            num_bits_per_symbol=self.config["num_bits_per_symbol"],
            constellation=self.constellation,
            hard_out=True,
        )

        # Resource grid mappers
        self.mapper_rg = ResourceGridMapper(self.resource_grid)
        self.remove_null = RemoveNulledSubcarriers(self.resource_grid)

        # Stream management for MIMO
        self.stream_management = sn.phy.mimo.StreamManagement(
            rx_tx_association=np.array([[1]]),
            num_streams_per_tx=self.config["num_streams_per_tx"],
        )

        self.demapper_rg = ResourceGridDemapper(
            resource_grid=self.resource_grid, stream_management=self.stream_management
        )

        # Channel estimator
        self.ls_estimator = LSChannelEstimator(
            resource_grid=self.resource_grid,
            interpolation_type="lin_time_avg",
        )

    def _create_channel_models(self):
        """Create channel models"""
        # TDL Models
        self.tdl1 = TDL(
            model="E",
            delay_spread=self.config["delay_spread"],
            carrier_frequency=self.config["carrier_frequency"],
            min_speed=0,
            max_speed=40,
            num_rx_ant=1,
            num_tx_ant=self.config["num_tx"],
        )

        # Other channel components
        self.channel_freq = ApplyOFDMChannel(add_awgn=False)
        self.awgn = AWGN()

    def calculate_power(self, signal):
        """Calculate signal power"""
        return tf.reduce_mean(tf.abs(signal.numpy()) ** 2).numpy()

    def normalize_signal(self, signal):
        """Normalize signal to unit power"""
        axis = np.arange(2, len(signal.shape))
        c = tf.reduce_mean(tf.square(tf.abs(signal)), axis=axis, keepdims=True)
        c = tf.complex(tf.sqrt(c), tf.constant(0.0, c.dtype))
        return tf.math.divide_no_nan(signal, c)

    def call(self, inputs):
        """
        Forward pass of the model (not used in this implementation)
        This is mainly a placeholder to satisfy the Keras Model interface
        """
        return inputs

    def generate_channel_data(self):
        """
        Generate channel data and signals based on configuration

        Returns:
            Tuple containing generated signals and parameters needed for simulation
        """
        # Generate channel realizations
        a1, tau1 = self.tdl1(
            batch_size=self.config["batch_size"],
            num_time_steps=self.resource_grid.num_ofdm_symbols,
            sampling_frequency=1 / self.resource_grid.ofdm_symbol_duration,
        )

        # Get subcarrier frequencies
        frequencies = subcarrier_frequencies(
            self.resource_grid.fft_size, self.resource_grid.subcarrier_spacing
        )

        # Convert channel impulse response to frequency domain
        h_freq = cir_to_ofdm_channel(frequencies, a1, tau1, normalize=True)

        # Generate binary data
        b1 = self.binary_source([
            self.config["batch_size"],
            self.config["num_tx"],
            self.config["num_streams_per_tx"],
            self.resource_grid.num_data_symbols * self.config["num_bits_per_symbol"],
        ])

        b2 = self.binary_source([
            self.config["batch_size"],
            self.config["num_tx"],
            self.config["num_streams_per_tx"],
            self.resource_grid.num_data_symbols * self.config["num_bits_per_symbol"],
        ])

        # Map binary data
        x1 = self.mapper(b1)
        x_rg1 = self.mapper_rg(x1)

        x2 = self.mapper(b2)
        x_rg2 = self.mapper_rg(x2)

        # Apply channel to first signal
        y1 = self.channel_freq(x_rg1, h_freq)

        # Create frequency-domain window
        h_freq_ifft = ifft(h_freq, axis=-1)
        condition_h_freq = tf.abs(h_freq_ifft[0, 0, 0, 0, 0, 0, :]) > 1
        channel_window = CustomWindow(
            length=self.resource_grid.fft_size,
            coefficients=condition_h_freq,
            trainable=False,
            normalize=False,
        )

        # Get resource grid parameters
        x_rg_n = self.remove_null(x_rg1)
        split_ind = self.resource_grid.dc_ind - self.resource_grid.num_guard_carriers[0]

        return {
            "b1": b1,
            "b2": b2,
            "x1": x1,
            "x2": x2,
            "x_rg1": x_rg1,
            "x_rg2": x_rg2,
            "y1": y1,
            "h_freq": h_freq,
            "frequencies": frequencies,
            "x_rg_n": x_rg_n,
            "split_ind": split_ind,
            "channel_window": channel_window,
        }

    def run_sicr_simulation(self, snr_values=None):
        """
        Run the SICR simulation for multiple SNR values

        Args:
            snr_values: Array of SNR values in dB

        Returns:
            Array of SICR values for each iteration and SNR
        """
        # Use default SNR values if none provided
        if snr_values is None:
            snr_values = np.arange(0, 61, 10)

        # Initialize arrays to store results
        sicr_results = np.zeros((self.config["iter"], len(snr_values)))

        # Generate channel data
        channel_data = self.generate_channel_data()

        # Extract data
        b2 = channel_data["b2"]
        x2 = channel_data["x2"]
        x_rg1 = channel_data["x_rg1"]
        x_rg2 = channel_data["x_rg2"]
        y1 = channel_data["y1"]
        h_freq = channel_data["h_freq"]
        frequencies = channel_data["frequencies"]
        x_rg_n = channel_data["x_rg_n"]
        split_ind = channel_data["split_ind"]
        channel_window = channel_data["channel_window"]

        # Normalize y1
        y1_norm = self.normalize_signal(y1)

        # Loop through SNR values
        for snr_idx, ebno_db in enumerate(snr_values):
            # Convert SNR to noise power
            no = ebnodb2no(
                ebno_db=ebno_db,
                num_bits_per_symbol=self.config["num_bits_per_symbol"],
                coderate=1,
                resource_grid=self.resource_grid,
            )

            no = tf.pow(10.0, -ebno_db / 10.0)

            # Apply channel and noise to second signal
            y2_won = self.channel_freq(x_rg2, h_freq)
            y2_wn = self.awgn(y2_won, no)

            # Apply delay
            phase = -2 * PI * frequencies * self.config["delay"]
            e_delay = tf.exp(tf.complex(tf.constant(0.0, dtype=tf.float32), phase))
            y2 = y2_wn * e_delay

            # Apply loopback amplification
            amp = tf.sqrt(db_to_lin(self.config["loopback_amplification"]))
            amp = tf.cast(amp, tf.complex64)
            y2_norm = self.normalize_signal(y2)
            y2_s = y2_norm / amp

            # Combine signals
            y_t = y1_norm + y2_s
            y_tp = y_t  # Initial signal for processing

            # Run iterations
            sicr_results[:, snr_idx] = self._run_sicr_iterations(
                y_t, y_tp, x_rg1, x_rg_n, x2, split_ind, channel_window, e_delay, no
            )

        return sicr_results

    def _run_sicr_iterations(
        self, y_t, y_tp, x_rg1, x_rg_n, x2, split_ind, channel_window, e_delay, no
    ):
        """
        Run the SICR algorithm iterations

        Args:
            y_t: Combined signal
            y_tp: Signal for processing
            x_rg1: First mapped resource grid
            x_rg_n: First mapped resource grid with nulls removed
            x2: Second mapped signal
            split_ind: Split index for DC carrier
            channel_window: Channel window for filtering
            e_delay: Delay exponential
            no: Noise power

        Returns:
            Array of SICR values for each iteration
        """
        sicr_values = np.zeros(self.config["iter"])

        # Perform iterations
        for i in range(self.config["iter"]):
            # Remove null subcarriers
            y_n = self.remove_null(y_tp)

            # Estimate channel
            lb_ls_est = tf.math.divide(y_n, x_rg_n)

            # Process through inner iterations for channel estimation
            lb_ls_fft = self._process_channel_estimation(
                lb_ls_est, split_ind, channel_window
            )

            # Calculate loopback estimate and residual
            lb_est = x_rg1 * lb_ls_fft
            residual_t = y_t - lb_est
            residual_offset = residual_t * tf.math.conj(e_delay)

            # Estimate forward channel
            fw_ls_est, _ = self.ls_estimator(residual_offset, no)

            # Process through inner iterations for interference estimation
            fw_ls_fft = self._process_channel_estimation(
                fw_ls_est, split_ind, channel_window
            )

            # Calculate equalized signal
            fw_eq = tf.reduce_mean(
                tf.reduce_mean(residual_offset / fw_ls_fft, axis=4), axis=3
            )
            fw_eq_wr = self.demapper_rg(fw_eq)

            # Calculate SNR and data
            noise2 = fw_eq_wr - x2
            snr = 10 * np.log10(self.calculate_power(x2) / self.calculate_power(noise2))
            no = tf.pow(10.0, -snr / 10.0)

            # Demap and remap
            data = self.demapper(fw_eq_wr, no)
            x_2r = self.mapper(data)
            x_rg_2r = self.mapper_rg(x_2r)

            # Reconstruct interference
            y2_2ro = tf.reduce_sum(tf.reduce_sum(x_rg_2r * fw_ls_fft, axis=4), axis=3)
            y2_2r = y2_2ro * e_delay

            # Calculate SICR
            sicr_value = 10 * np.log10(
                self.calculate_power(y_t - y2_2r)
                / self.calculate_power(residual_t - y2_2r)
            )
            sicr_values[i] = sicr_value

            # Update signal for next iteration
            y_tp = y_t - y2_2r

        return sicr_values

    def _process_channel_estimation(self, ls_est, split_ind, channel_window):
        """
        Process interference estimation through inner iterations

        Args:
            ls_est: Forward channel least squares estimate
            split_ind: Split index for DC carrier
            channel_window: Window for time domain filtering

        Returns:
            Processed interference estimate in frequency domain
        """
        # Create parts for concatenation
        first_part = tf.repeat(ls_est[..., 0:1], repeats=639, axis=-1)
        last_part = tf.repeat(ls_est[..., -1:], repeats=639, axis=-1)
        dc_part = ls_est[..., 3456:3457]  # Specific to the frequency configuration

        for k in range(self.config["iter_n2"]):
            # Concatenate parts
            fw_concat = tf.concat(
                [
                    first_part,
                    ls_est[..., :split_ind],
                    dc_part,
                    ls_est[..., split_ind:],
                    last_part,
                ],
                axis=-1,
            )

            # FFT processing with window
            fw_ifft = ifft(fw_concat)
            fw_win = channel_window(fw_ifft)
            fw_fft = fft(fw_win)

            # Update parts for next iteration
            first_part = fw_fft[..., 0:639]
            last_part = fw_fft[..., -639:]

        return fw_fft

    def plot_results(self, results, snr_values, save_path=None, show_plot=True):
        """
        Plot SICR results

        Args:
            results: SICR results from run_sicr_simulation
            snr_values: SNR values used in simulation
            save_path: Path to save the plots (None = don't save)
            show_plot: Whether to display the plot interactively

        Returns:
            Tuple of figure objects (fig1, fig2)
        """
        # Use non-interactive backend if not showing plot
        if not show_plot:
            plt.switch_backend("Agg")

        # Create figure for SNR vs SICR
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        for i in range(self.config["iter"]):
            ax1.plot(snr_values, results[i], marker="o", label=f"Iteration {i}")
        ax1.set_xlabel("SNR (dB)", fontsize=14)
        ax1.set_ylabel("SICR (dB)", fontsize=14)
        ax1.set_title("Signal to Interference Cancellation Ratio vs SNR", fontsize=15)
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.7)

        # Create figure for Iteration vs SICR
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        iteration_values = np.arange(0, self.config["iter"])
        for snr_idx, snr_value in enumerate(snr_values):
            ax2.plot(
                iteration_values,
                results[:, snr_idx],
                marker="o",
                label=f"SNR = {snr_value} dB",
            )
        ax2.set_xlabel("Iteration", fontsize=14)
        ax2.set_ylabel("SICR (dB)", fontsize=14)
        ax2.set_title(
            "Signal to Interference Cancellation Ratio vs Iteration", fontsize=15
        )
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.7)

        # Save plots if path is provided
        if save_path:
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Save both plots
            fig1.savefig(f"{save_path}_vs_snr.png", dpi=300, bbox_inches="tight")
            fig2.savefig(f"{save_path}_vs_iterations.png", dpi=300, bbox_inches="tight")

        # Show plots if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig1)
            plt.close(fig2)

        return fig1, fig2


# Test code
if __name__ == "__main__":
    # Suppress TensorFlow logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    print("Testing SICR Keras Model")
    print("------------------------")

    # Create output directory for plots
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a model instance with smaller parameters for faster testing
    print("\n1. Creating model with reduced parameters for faster execution...")

    # Configuration with reduced parameters for faster testing
    config = {
        "batch_size": 1,
        "iter": 8,  # Reduced iterations
        "iter_n": 8,  # Reduced inner iterations
        "iter_n2": 8,  # Reduced inner iterations
        "fft_size": 8192,  # Keep the same FFT size
        "num_useful_subcarriers": 6913,  # Keep the same number of subcarriers
        "seed": 42,
    }

    model = SICRKerasModel(config)
    print("   Model created successfully")

    # Run simulation with fewer SNR values
    print("\n2. Running SICR simulation...")
    snr_values = np.array([
        0,
        10,
        20,
        30,
        40,
        50,
        60,
    ])  # Using fewer SNR points for quicker testing
    results = model.run_sicr_simulation(snr_values)
    print("   Simulation completed")
    print(f"   Results shape: {results.shape}")

    # Show simulation results
    print("\n3. SICR Results (dB):")
    print("   SNR (dB) |", end="")
    for snr in snr_values:
        print(f" {snr:6.1f} |", end="")
    print()

    print("   ---------+", end="")
    for _ in snr_values:
        print("--------+", end="")
    print()

    for i in range(config["iter"]):
        print(f"   Iter {i:2d}  |", end="")
        for j in range(len(snr_values)):
            print(f" {results[i, j]:6.2f} |", end="")
        print()

    # Generate plots
    print("\n4. Generating plots...")
    model.plot_results(
        results, snr_values, save_path=f"{output_dir}/sicr_keras_test", show_plot=False
    )
    print(
        f"   Plots saved to {output_dir}/sicr_keras_test_vs_snr.png and {output_dir}/sicr_keras_test_vs_iterations.png"
    )

    print("\nTest completed successfully")
