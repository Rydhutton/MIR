#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import wave
import os
import struct
import matplotlib.pyplot as plt
import math
import random
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

    # PART 1 ***********************

    input = Signal()
    input.wav_read('violin.wav')
    compSpectr = [0] * len(input.data)
    compSpectr = np.fft.fft(input.data) / len(input.data)
    timeDm = np.fft.ifft(compSpectr) * len(compSpectr)
    output = Signal(44100, None, timeDm)
    output.wav_write('violinTransform.wav')

    # PART 2 ************************

    input = Signal()
    input.wav_read('violin.wav')
    compSpectra = [0] * len(input.data)
    compSpectra = np.fft.fft(input.data) / len(input.data)

    for x in range(len(compSpectra)):
        magnitude = np.abs(compSpectra[x])
        val = random.uniform(0.0, 1.0) * 2 * math.pi
        compSpectra[x] = magnitude * (math.cos(val) + (1j * math.sin(val)))

    timeDom = np.fft.ifft(compSpectra) * len(compSpectra)
    output = Signal(44100, None, timeDom)
    output.wav_write('violinTransform2.wav')

    # PART 3 *************************

    x1 = 0.0
    x2 = 0.0
    x3 = 0.0
    x4 = 0.0

    input2 = Signal()
    input2.wav_read('violin.wav')
    compSpectra2 = [0] * len(input2.data)
    compSpectra2 = np.fft.fft(input2.data) / len(input2.data)

    for x in range(len(compSpectra2)):
        magnitude2 = np.abs(compSpectra2[x])
        val2 = random.uniform(0.0, 1.0) * 2 * math.pi
        compSpectra2[x] = magnitude2 * (math.cos(val2) + (1j * math.sin(val2)))
        val = compSpectra2[x]
        if val > x1:
            x4 = x3
            x3 = x2
            x2 = x1
            x1 = val
        elif val > x2:
            x4 = x3
            x3 = x2
            x2 = val
        elif val > x3:
            x4 = x3
            x3 = val
        elif val > x4:
            x4 = val
        else:
            compSpectra2[x] = 0

    print "4 highest magnitudes were: ", x1, x2, x3, x4

    timeDom2 = np.fft.ifft(compSpectra2) * len(compSpectra2)
    output2 = Signal(44100, None, timeDom2)
    output2.wav_write('violinTransform3.wav')


if __name__ == "__main__":
    main()


