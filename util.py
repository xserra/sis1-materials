import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import librosa
from plotly.offline import iplot
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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


def load_audio(filepath):
    data, sr = sf.read(filepath)

    # Convert to mono
    if len(data.shape) > 1:
        data = data[:, 0]

    # Remove DC component
    data = data - np.mean(data)

    return data, sr


def save_audio(filepath, data, samplerate):
    sf.write(filepath, data, samplerate)


def plot_signals(y, sr, t_start=0, t_end=-1, name='audio signal', mode='lines'):
    if type(y) is not list:
        y = [y]

    if type(name) is not list:
        names = [name + ' ' + str(j) for j in range(len(y))]
    else:
        assert len(name) == len(y)
        names = name

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
    iplot(data_plot)


def plot_spectrogram(ff, tt, S):
    S = librosa.power_to_db(S)
    plt.pcolormesh(tt, ff, S, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')


def plot_mean_spectrogram(S, sr, n_fft):
    freqs = np.linspace(0, sr/2 - sr/n_fft, int(n_fft/2+1))

    if type(S) is not list:
        mean_spec = np.mean(S, axis=1)

        data_plot = [
            go.Scatter(
                x=freqs,
                y=np.sqrt(mean_spec/np.amax(mean_spec)),
                mode='lines+markers',
                line=dict(shape='linear', color=colors[0])
            )
        ]
    else:
        data_plot = []
        for j in range(len(S)):
            mean_spec = np.mean(S[j], axis=1)
            data_plot.append(
                go.Scatter(
                    x=freqs,
                    y=np.sqrt(mean_spec/np.amax(mean_spec)),
                    mode='lines+markers',
                    name=str(j),
                    line=dict(shape='linear', color=colors[j % len(colors)])
                )
            )
    fig = go.Figure(data_plot)
    fig.update_yaxes(type="log")
    fig.show()


def plot_spectrum_at(ff, tt, S, time):
    if type(S) is not list:
        time_hop = np.argmin((tt - time)**2)

        fft = S[:, time_hop]

        data_plot = [
            go.Scatter(
                x=ff,
                y=np.sqrt(fft/np.amax(fft)),
                mode='lines+markers',
                line=dict(shape='linear', color=colors[0])
            )
        ]

    else:
        data_plot = []
        time_hop = np.argmin((tt - time)**2)
        for j in range(len(S)):
            fft = S[j][:, time_hop]
            data_plot.append(
                go.Scatter(
                    x=ff,
                    y=np.sqrt(fft/np.amax(fft)),
                    mode='lines+markers',
                    name=str(j),
                    line=dict(shape='linear', color=colors[j % len(colors)])
                )
            )
    fig = go.Figure(data_plot)
    fig.update_yaxes(type="log")
    fig.show()


def plot_complex(z, name='z'):
    if type(z) is not list:
        z = [z]

    if type(name) is not list:
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


def plot_frequency_response(w, H):
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(
        go.Scatter(x=w, y=np.abs(H), name="Magnitude"),
    row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=w, y=np.angle(H), name='Phase'),
        row=2, col=1
    )

    fig.update_xaxes(
        title_text="Normalized Radian Frequency", row=2, col=1,
        tickmode = 'array',
        tickvals = [-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4,np.pi],
        ticktext = ['$-\pi$', '$-3\pi/4$', '$-\pi/2$', '$-\pi/4$', '$0$', '$\pi/4$', '$\pi/2$', '$3\pi/4$', '$\pi$']
    )
    fig.update_xaxes(
        row=1, col=1,
        tickmode = 'array',
        tickvals = [-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4,np.pi],
        ticktext = ['$-\pi$', '$-3\pi/4$', '$-\pi/2$', '$-\pi/4$', '$0$', '$\pi/4$', '$\pi/2$', '$3\pi/4$', '$\pi$']
    )
    fig.update_yaxes(title_text="Phase <H", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude |H|", row=1, col=1)
    fig.update_layout(
        title="Frequency Response",
    )
    fig.show()


def plot_zeros_poles(z, p):
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
