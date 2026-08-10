from scipy import signal
import numpy as np

b = np.array([1, 2, 1])
w, h = signal.freqz(b)
fc = np.pi / 3

import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(tight_layout=True)
ax1.set_title("Frequency Response of FIR Filter")
ax1.plot(w, 20 * np.log10(abs(h)), 'C0')
ax1.set_ylabel("Amplitude in dB", color='C0')
# ax1.plot(w, abs(h), 'C0')
# ax1.set_ylabel("Amplitude", color='C0')
ax1.set(xlabel="Frequency in rad/sample", xlim=(0, np.pi))
ax1.axvline(fc, color='black', linestyle=':', linewidth=0.8)

ax2 = ax1.twinx()
phase = np.unwrap(np.angle(h))
ax2.plot(w, phase, 'C1')
ax2.set_ylabel('Phase [rad]', color='C1')
ax2.grid(True)
ax2.axis('tight')
plt.show()
