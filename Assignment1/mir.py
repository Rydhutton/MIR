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

    sin1 = Sinusoid(1, 256, 1.0, 100, 0)
    sin2 = Sinusoid(1, 256, 1.5, 200, 1)
    sin3 = Sinusoid(1, 256, 2.0, 300, 2)

    signal = Mixture(sin1, sin2, sin3)

    output = []
    mag = []
    phase = []
    input = signal.data
    length = len(signal.data)

    # my DFT
    for n in range(length):
        real = 0.0
        imag = 0.0
        for r in range(length):
            real += (input[r] * math.cos(2 * math.pi * n * r / length))
            imag += (-input[r] * math.sin(2 * math.pi * n * r / length))
        mag.append(math.sqrt(real**2 + imag**2) / length)
        phase.append(math.atan((imag / real) / length))
        output.append(real)
        output.append(imag)

    # numpy DFT using FFT
    comp = fft(signal.data)
    compMag = np.abs(comp)
    norm = len(compMag)
    for x in range(len(compMag)):
        compMag[x] = compMag[x] / norm #normalize

    plt.plot(mag)
    plt.title("My Code")
    plt.show()

    plt.plot(compMag)
    plt.title("Numpy Package")
    plt.show()

    plt.plot(phase)
    plt.title('Phase Spectrum')
    plt.show()

    # Part 3 ***************************

    samples = 256.0       # Num of samples
    frequency = 4       # The bin
    sin = [0.0] * 256
    cos = [0.0] * 256

    for x in range(256):
        cos[x] = (math.cos((x / samples) * 2 * math.pi * frequency))
        sin[x] = (math.sin((x / samples) * 2 * math.pi * frequency))

    plt.plot(cos)
    plt.title('Cos, bin 4')
    plt.show()

    plt.plot(sin)
    plt.title('Sin, bin 4')
    plt.show()

    # Part 4 ****************************

    sinA = Sinusoid(1, 256, 1.0, 4, 0)
    sinB = Sinusoid(1, 256, 1.5, 8, 1)
    sinC = Sinusoid(1, 256, 2.0, 12, 2)

    signalO = Mixture(sinA, sinB, sinC)

    plt.plot(signalO.data)
    plt.title('3-Harmonic Sinusoids Mixture')
    plt.show()

    sinR = [0.0] * 256
    cosR = [0.0] * 256
    signalOut = [0.0] * 256
    signalOutR = [0.0] * 256

    frequency = 64
    for x in range(256):
        cosR[x] = (math.cos((x / samples) * 2 * math.pi * frequency))
        sinR[x] = (math.sin((x / samples) * 2 * math.pi * frequency))

    for x in range(int(samples)):
        signalOut[x] = sin[x] * signalO.data[x]
        signalOutR[x] = sinR[x] * signalO.data[x]

    plt.plot(signalOut)
    plt.title('Multiplied w/ Basis Sine Bin 4')
    plt.show()

    plt.plot(signalOutR)
    plt.title('Multiplied w/ Basis Sine Bin 64')
    plt.show()

    for x in range(int(samples)):
        signalOut[x] = cos[x] * signalO.data[x]
        signalOutR[x] = cosR[x] * signalO.data[x]

    plt.plot(signalOut)
    plt.title('Multiplied w/ Basis Cosine Bin 4')
    plt.show()

    plt.plot(signalOutR)
    plt.title('Multiplied w/ Basis Cosine Bin 64')
    plt.show()

if __name__ == "__main__":
    main()
