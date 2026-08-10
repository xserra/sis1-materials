"""Shared audio, plotting, and signal-processing helpers used across the labs."""

import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from plotly.offline import iplot
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy import signal

colors = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf'
]

def plot_complex(z, name='z'):
    """Plot complex numbers on the complex plane together with the unit circle."""
    if not isinstance(z, list):
        z = [z]

    if not isinstance(name, list):
        names = [name + '_' + str(j) for j in range(len(z))]
    else:
        assert len(name) == len(z)
        names = name

    omega = np.linspace(0, 2*np.pi, 1000)
    data_plot = [
        go.Scatter(
            x=np.cos(omega),
            y=np.sin(omega),
            mode='lines',
            name='unit circle',
            line=dict(shape='linear', color='rgb(150,150,150)', dash='dash')
        )
    ]
    arrows = []
    for i, z_i in enumerate(z):
        data_plot.append(
            go.Scatter(
                x=[np.real(z_i)], y=[np.imag(z_i)],
                mode='markers',
                name=names[i],
                marker={'color': colors[i % len(colors)]}
            )
        )
        arrows.append(
            go.layout.Annotation(dict(
                x=np.real(z_i),
                y=np.imag(z_i),
                showarrow=True,
                axref="x", ayref='y',
                text="",
                ax=0,
                ay=0,
                arrowhead=3,
                arrowwidth=1.5,
                arrowcolor=colors[i % len(colors)],)
            )
        )

    fig = go.Figure(data_plot)
    fig.update_layout(
        xaxis_title="Real",
        yaxis_title="Imaginary",
    )

    fig.update_layout(annotations=arrows)

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.show()

def load_audio(filepath):
    """Load an audio file as mono, remove its DC component, and return data and sample rate."""
    data, sr = sf.read(filepath)

    # Convert to mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Remove DC component
    data = data - np.mean(data)

    return data, sr


def save_audio(filepath, data, samplerate):
    """Write audio data to disk at the given sample rate."""
    sf.write(filepath, data, samplerate)


def plot_signals(y, sr, t_start=0, t_end=-1, name='audio signal', mode='lines'):
    """Plot one signal or a list of signals over time using Plotly."""
    if not isinstance(y, list):
        y = [y]

    names = name
    if not isinstance(names, list):
        names = [name + ' ' + str(j) for j in range(len(y))]

    Ts = 1/sr

    t = np.linspace(0, len(y[0])*Ts, len(y[0]))

    if t_end == -1:
        t_end = len(y[0])*Ts

    samples_start = int(t_start*sr)
    samples_end = int(t_end*sr)

    data_plot = []
    for j in range(len(y)):
        data_plot.append(
            go.Scatter(
                x=t[samples_start:samples_end],
                y=y[j][samples_start:samples_end],
                name=names[j],
                mode=mode,
                line=dict(shape='linear', color=colors[j % len(colors)])
            )
        )
    fig = go.Figure(data=data_plot)
    fig.show()

def plot_spectrum(x: np.ndarray, w: np.ndarray | None = None, N: int | None = None, sr: float | None = None):
    """Plot the magnitude spectrum of a signal in decibels and return the axis.

    Args:
        x: Input signal.
        w: Analysis window. If omitted, a Hann window is used by default.
        N: FFT size. It must match the signal length; zero-padding is not used.
        sr: Sampling rate in Hz.
    """
    if sr is None:
        raise ValueError('sr must be provided')

    x = np.asarray(x)
    if w is None:
        w = np.hanning(x.size)
    else:
        w = np.asarray(w, dtype=float)
        if w.size != x.size:
            raise ValueError('w must have the same length as x')

    if N is None:
        N = x.size
    elif N != x.size:
        raise ValueError('N must match the signal length; zero-padding is not used')

    w = w / np.sum(w)
    xw = x * w
    Xh = np.fft.rfft(xw, n=N)
    freqs = np.fft.rfftfreq(N, d=1 / sr)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(Xh), 1e-12))

    plt.figure(figsize=(10, 3))
    ax = plt.gca()
    ax.plot(freqs, magnitude_db)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Magnitude spectrum')
    return ax



def plot_spectrogram(x: np.ndarray, sr: float, w: np.ndarray | None = None, N: int | None = None, H: int = 256):
    """Plot the magnitude spectrogram of a signal in decibels and return the axis.

    Args:
        x: Input signal.
        sr: Sampling rate in Hz.
        w: Analysis window. If omitted, a Hann window is used by default.
        N: FFT size. If omitted, the window length is used.
        H: Hop size in samples.
    """
    if H <= 0:
        raise ValueError(f'Hop size (H={H}) must be positive')

    if w is None:
        w = 'hann'

    if N is None:
        if isinstance(w, str):
            N = 256 # A default for string window
        else:
            N = np.asarray(w).size

    freqs, times, magnitude_spectra = signal.stft(x, fs=sr, window=w, nperseg=N, noverlap=N-H, nfft=N)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(magnitude_spectra), 1e-12))

    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.pcolormesh(times, freqs, magnitude_db, shading='gouraud')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_title('Magnitude spectrogram')
    return ax


def plot_frequency_response(b, a=1, worN=2048, sr=None):
    """Plot the magnitude and phase response of a discrete-time filter."""
    w, H = signal.freqz(b, a, worN=worN)

    if sr is None:
        freqs = w / np.pi
        xlabel = 'Normalized frequency (×π rad/sample)'
    else:
        freqs = w * sr / (2 * np.pi)
        xlabel = 'Frequency (Hz)'

    magnitude_db = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
    phase = np.angle(H)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(freqs, magnitude_db)
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title('Frequency response')
    axes[0].grid(alpha=0.3)

    axes[1].plot(freqs, phase)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('Phase (rad)')
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    return axes

def plot_zeros_poles(z, p):
    """Plot zeros and poles on the complex plane together with the unit circle."""
    z = np.asarray(z)
    p = np.asarray(p)

    omega = np.linspace(0, 2*np.pi, 1000)

    data_plot = [
        go.Scatter(
            x=np.cos(omega),
            y=np.sin(omega),
            mode='lines',
            name='unit circle',
            line=dict(shape='linear', color='rgb(150,150,150)', dash='dash')
        )
    ]

    data_plot.append(
        go.Scatter(
            x=np.real(z), y=np.imag(z),
            mode='markers',
            name = 'zeros',
            marker={
              'color': colors[0],
              'symbol': 'circle',
              'size': 14
            }
        )
    )

    data_plot.append(
        go.Scatter(
            x=np.real(p), y=np.imag(p),
            mode='markers',
            name = 'poles',
            marker={
              'color': colors[1],
              'symbol': 'x',
              'size': 14
            }
        )
    )

    fig = go.Figure(data_plot)
    fig.update_layout(
        xaxis_title="Real",
        yaxis_title="Imaginary",
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.show()
