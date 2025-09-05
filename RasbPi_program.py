import time
import numpy as np
import matplotlib.pyplot as plt

import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from board import SCL, SDA

# Setup I2C
i2c = busio.I2C(SCL, SDA, frequency=100000)
i2c._i2c._bus = 21  # Force i2c-21 usage

# ADC Setup
ads = ADS.ADS1115(i2c)
chan = AnalogIn(ads, ADS.P0)

# --- Sampling Configuration ---
num_samples = 2000
sampling_interval = 0.001  # 1 ms
Fs = 1 / sampling_interval

# --- Initialize Time-Domain Plot ---
plt.ion()
fig, ax_time = plt.subplots(figsize=(10, 4))
line_time, = ax_time.plot([], [], lw=1.5)
ax_time.set_xlim(0, num_samples * sampling_interval)
ax_time.set_ylim(0, 5)
ax_time.set_title("Live Oscilloscope (Time Domain)")
ax_time.set_xlabel("Time (s)")
ax_time.set_ylabel("Voltage (V)")
ax_time.grid(True)

# --- Data Collection ---
samples = []
timestamps = []

print("Collecting and plotting time-domain data...")
start_time = time.time()

try:
    for i in range(num_samples):
        voltage = chan.voltage
        current_time = time.time() - start_time
        samples.append(voltage)
        timestamps.append(current_time)

        # Update plot every 10 samples
        if i % 10 == 0:
            line_time.set_data(timestamps, samples)
            ax_time.set_xlim(max(0, current_time - 2), current_time + 0.1)
            ax_time.set_ylim(min(samples[-200:]) - 0.1, max(samples[-200:]) + 0.1)
            plt.pause(0.001)

        time.sleep(sampling_interval)

except KeyboardInterrupt:
    print("Interrupted.")

# Final time-domain freeze plot
plt.ioff()
line_time.set_data(timestamps, samples)
ax_time.set_xlim(0, timestamps[-1])
ax_time.set_ylim(min(samples) - 0.1, max(samples) + 0.1)
plt.tight_layout()
plt.show()

# --- Frequency-Domain Analysis ---
print("Calculating FFT and displaying frequency-domain plot...")
samples_np = np.array(samples)
samples_np -= np.mean(samples_np)  # Remove DC component
windowed = samples_np * np.hanning(len(samples_np))
fft_vals = np.fft.fft(windowed)
fft_freqs = np.fft.fftfreq(num_samples, d=sampling_interval)
positive_mask = fft_freqs >= 0
freqs = fft_freqs[positive_mask]
amplitudes = (2.0 / num_samples) * np.abs(fft_vals[positive_mask])

# --- Frequency-Domain Plot ---
plt.figure(figsize=(10, 4))
plt.plot(freqs, amplitudes, color='darkgreen')
plt.title("Frequency Spectrum (FFT)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 500)  # Up to Nyquist (500 Hz)
plt.grid(True)
plt.tight_layout()
plt.show()
