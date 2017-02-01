#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import wave
import os
import struct
import matplotlib.pyplot as plt
import math
from scipy.fftpack import fft, fftshift



def np_to_int16_bytes(x):
    x = np.int16(x * 2**(16-1))
    return struct.pack(str(len(x))+'h', *x)


def int16_bytes_to_np(x, num, sw):
    x = np.array(struct.unpack(str(num)+'h', x))
    x = x.astype(np.float) / 2**(16-1)
    return x


class Signal(object):

    def __init__(self, Fs=44100, duration=None, data=[]):
        self._duration = duration
        self._Fs, self._Ts = Fs, 1./Fs
        self._data = data
        if duration:
            self._n = np.arange(0, duration, self._Ts)

    def plot(self):
        if len(self._data):
            plt.plot(self._n, self._data)
            plt.show()

    def estimate_amplitude(self, base):
        amp = (np.dot(self._data, base._data) /
               np.dot(base._data, base._data) )
        return amp

    def dft(self):
        if len(self._data):
            return np.fft.fft(self._data)/len(self._data)

    def wav_write(self, outfile, Nch=1, Sw=2, normalize=True):
        if len(self._data):
            x = self._data
            x = x / max(x) if normalize else x
            dst = wave.open(outfile, 'wb')
            dst.setparams((Nch, Sw, self._Fs, len(x), 'NONE', 'not_compressed'))
            dst.writeframes(np_to_int16_bytes(x))
            dst.close()

    def wav_read(self, in_file):
        assert(os.path.exists(in_file))
        src = wave.open(in_file, 'rb')
        nch, sw, fs, nframes, _, _ = src.getparams()
        self.__init__(Fs=fs, duration=nframes/fs)
        assert(nch == 1), "wav must be 1 ch"
        self._data = int16_bytes_to_np(src.readframes(nframes), nframes, sw)
        src.close()

    @property
    def data(self):
        return self._data

    @property
    def duration(self):
        return self._duration

    @property
    def params(self):
        return self._duration, self._Fs, self._Ts



class Sinusoid(Signal):

    def __init__(self, duration=1, Fs=44100.0, amp=1.0, freq=440.0, phase=0):
        super(self.__class__, self).__init__(duration=duration, Fs=Fs)
        self.A, self.f, self.phi = amp, freq, phase
        self._w = 2 * np.pi * self.f
        self.__make()

    def __make(self):
        self._data = self.A * np.sin(self._w * self._n + self.phi)

    def power(self):
        return self.A**2/2.0

    def add_noise(self, snr):
        sigma2 = self.power()/(10**(snr/10.0))
        noise = np.random.normal(0, np.sqrt(sigma2), len(self._data))
        self._data += noise

    def remove_noise(self):
        self.__make()

    def shift(self, phi):
        self._phi = phi
        self.__make()


class Mixture(Signal):

    def __init__(self, *signals):
        duration, fs, _ = signals[0].params
        super(self.__class__, self).__init__(duration=duration, Fs=fs)
        self._data = np.zeros(len(signals[0].data))
        for sig in signals:
            self._data += sig.data


class Sequence(Signal):

    def __init__(self, *signals):
        _, fs, _ = signals[0].params
        duration = np.sum([sig.duration for sig in signals])
        super(self.__class__, self).__init__(duration=duration, Fs=fs)
        self._data = np.hstack([sig.data for sig in signals])


class MidiNotes():
    TUNING = 440
    NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
             'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self):
        self._notes = {}
        for i in range(24, 109):
            f = 2**((i-69)/12.0) * self.TUNING
            self._notes[self.NAMES[i % 12] + str(i/12-1)] = f

    def freq(self, name):
        return self._notes[name]


def main():

    # PART 1 ****************************

    samples = 256.0
    freqs = [4, 8, 12]
    amps = [1.0, 1.5, 2.0]
    duration = 1
    data = [0.0] * int(samples)

    for x in range(int(samples)):
        count = 0
        for y in range(len(freqs)):
            count += math.sin(((x / samples) * 2 * math.pi) * freqs[y] * amps[y])
        data[x] = count

    signalFFT = Signal(samples, duration, data)
    signalFFT.plot()

    # PART 2 ****************************

    samples2 = 256.0
    freqs2 = [527, 1053, 1578, 133, 2109, 3200, 2639, 3746, 4305, 4700]
    decibles = [-22.1, -26.0, -31.1, -32.5, -36.2, -37.2, -38.5, -43.6, -41.0, -47.7]
    duration2 = 0.19
    data2 = [0.0] * int(samples)
    sampleRate = int(samples2 * duration2)

    for x in range(int(samples2)):
        count = 0
        for y in range(len(freqs2)):
            count += (10**(decibles[y]/10)) * math.sin((freqs2[y]/sampleRate) * (x/samples2) * 2 * math.pi)
        data2[x] = count

    plt.plot(data2)
    plt.title('Synthesized Piano & Guitar C4')
    plt.show()

    timeDom2 = np.fft.ifft(data2) * len(data2)
    output2 = Signal(44100, None, timeDom2)
    output2.wav_write('Synth.wav')

    # PART 3 *****************************

    sampleRate3 = 1024 # samples per second
    note_duration = 0.25
    notes = []

if __name__ == "__main__":
    main()


