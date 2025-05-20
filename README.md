# SICR - Signal to Interference Cancellation Ratio

This repository is a personal implementation based on:

Z. H. Hong et al., "Iterative Successive Nonlinear Self-Interference Cancellation for In-Band Full-Duplex Communications," in IEEE Transactions on Broadcasting, vol. 70, no. 1, pp. 2-13, March 2024, doi: 10.1109/TBC.2023.3291136.

## Overview

The implementation uses [Sionna](https://nvlabs.github.io/sionna/), an open-source library for simulating physical layer communications built on TensorFlow. The code simulates an in-band full-duplex communication system and analyzes the Signal to Interference Cancellation Ratio (SICR) across various signal-to-noise ratios and iterations.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Sionna
- NumPy
- Matplotlib (for visualization)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ale1989/sicr.git
   cd sicr
   ```

2. Install the required packages:
   ```bash
   pip install sionna
   ```

## Project Structure

- [`sicr_keras_m.py`](sicr_keras_m.py ) - Main implementation of the SICR model using Keras
- [`tdl.py`](tdl.py ) - Tapped Delay Line channel model for read TDL-E.json
- [`TDL-E.json`](TDL-E.json ) - Parameters for TU-6 using the same dictionary structure from sionna ;)

## Usage

Run the main simulation:

```bash
python sicr_keras_m.py
```

By default, this will:
1. Create an SICR model with pre-configured parameters
2. Run the simulation across several SNR values (0-60dB)
3. Calculate SICR for each iteration and SNR value
4. Generate plots showing the results
5. Save the plots in a `plots` directory

## Customizing Simulation

You can customize the simulation by modifying the configuration parameters in [`sicr_keras_m.py`](sicr_keras_m.py ):

```python
config = {
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
    "iter": 7,  # Number of iterations
    "iter_n": 8,  # Inner iterations
    "iter_n2": 8,  # Inner iterations for channel estimation
    "delay": 11e-6,  # Delay parameter
}

model = SICRKerasModel(config)
```

## Results

The simulation generates two plots:
1. SICR vs SNR for each iteration
2. SICR vs Iteration for each SNR value

These visualizations help understand the effectiveness of the iterative self-interference cancellation technique at different signal quality levels.

## License

Please respect the license of the original paper and the Sionna library.
