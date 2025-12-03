import sar_funcs as sar
import numpy as np
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def main():
    Fs = 23.328e9 # Sampling frequency

    fc = 7.25e9
    BW = 1.5e9
    frac_bw = BW/fc
    PRF = 14e6
    VTX = 0.02


    # pulse = sar.generate_uwb_pulse(Fs, pulse_length, fc, frac_bw)
    # print(pulse)
    # print(pulse.shape)

    dt = 1/Fs/10
    T = (1/PRF)/20
    t = np.arange(-T/2, T/2, dt)

    i, q, e = signal.gausspulse(t, fc, bw=frac_bw, retquad=True, retenv=True)
    # i, q, e = signal.gausspulse(t, fc=7.5, retquad=True, retenv=True)

    # plt.plot(t, i, t, q, t, e, '--')
    # plt.plot(t, i, t, q, '--')
    # plt.show()

    t, pulse = sar.generate_uwb_pulse(Fs, fc, frac_bw, PRF, VTX)
    # plt.plot(t, pulse)
    # plt.show()

    data = np.random.randn(10, 100)
    r_min = 0
    r_max = 10
    corr = sar.pulse_compression2(data, pulse, r_min, r_max, Fs)
    print(corr)
    print(corr.shape)

    data = np.random.randn(100)
    r_min = 0
    r_max = 10
    corr = sar.pulse_compression(data, pulse)
    print(corr)
    print(corr.shape)

if __name__ == '__main__':
    main()
