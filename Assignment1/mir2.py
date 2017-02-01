#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import wave
import os
import struct
import matplotlib.pyplot as plt
import math
from scipy.fftpack import fft, fftshift
from scipy.io import wavfile



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

from scipy import signal

def main():

    # QUESTION 2 ********************************************

    # PART 1

    N = 2048         #DFT Magnitude size, ie num of samplepoints

    sin1 = Sinusoid(1, N, 1.0, 525.0, 0)    #Sinusoid 1 with Bin 525.0
    sin2 = Sinusoid(1, N, 1.0, 600.5, 0)    #Sinusoid 2 with Bin 600.5
    window = np.hanning(N)                  #Hamming Window

    s1FFT = fft(sin1.data)
    s2FFT = fft(sin2.data)
    mag1 = np.abs(s1FFT)
    norm1 = len(mag1)
    for x in range(len(mag1)):
        mag1[x] = mag1[x] / norm1
        mag1[x] = 10*math.log10(mag1[x])
    mag2 = np.abs(s2FFT)
    norm2 = len(mag2)
    for x in range(len(mag2)):
        mag2[x] = mag2[x] / norm2
        mag2[x] = 10*math.log10(mag2[x])

    data = [0.0] * int(N)
    for x in range(int(N)):
        data[x] = window[x] * sin1.data[x]
    dataFFT = fft(data)
    magData = np.abs(dataFFT)
    normData = len(magData)
    for x in range(normData):
        magData[x] = magData[x] / normData
        magData[x] = 10*np.log10(magData[x])

    plt.plot(mag1, 'b', label='w/o H.W.')
    plt.plot(magData, 'r', label='w/ H.W.')
    plt.legend()
    plt.title('Bin 525 Sine Wave')
    plt.show()

    plt.plot(mag2, 'b')
    plt.title('Bin 600.5 Sine Wave')
    plt.show()

    # PART 2

    window2048 = np.hamming(2048)

    #fs, data2 = wavfile.read('flute.wav')

    fs, dataR = wavfile.read('CSC475.wav')
    data2 = dataR.T[0]

    data2048 = [0.0] * N
    data2048HW = [0.0] * N


    for x in range(N):
        data2048[x] = data2[x]
        data2048HW[x] = data2[x] * window2048[x]

    #test = abs(np.fft.rfft(data2048HW))
    data2048FFT = fft(data2048)
    data2048HWFFT = fft(data2048HW)
    magData2048 = np.abs(data2048FFT)
    magData2048HW = np.abs(data2048HWFFT)
    normData1 = len(magData2048)
    normData2 = len(magData2048HW)

    for x in range(normData1):
        magData2048[x] = magData2048[x] / normData1
        magData2048[x] = 10*np.log10(magData2048[x])
    for x in range(normData2):
        magData2048HW[x] = magData2048HW[x] / normData2
        magData2048HW[x] = 10*np.log10(magData2048HW[x])

    plt.plot(magData2048, 'b', label='w/o H.W.')
    plt.legend()
    plt.title('Flute E note')
    #plt.show()
    plt.plot(magData2048HW, 'r', label='w/ H.W.')
    plt.legend()
    plt.title('Flute E note')
    plt.show()


if __name__ == "__main__":
    main()


