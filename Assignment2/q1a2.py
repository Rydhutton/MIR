import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.fftpack import fft, fftshift, ifft
import wave
import os
import struct
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


def f0max(name):

    # Part 1

    input = Signal()
    input.wav_read(name)
    data = input.data

    size = int(len(data) / 2048)
    chunk = 2048
    currChunk = [0.0] * chunk
    f0List = []

    while chunk < 2048 * size:
        counter = 0
        for x in range(chunk - 2048, chunk):
            currChunk[counter] = data[x]
            counter += 1
        chunk += 2048
        compSpectra = fft(currChunk)
        magnitude = np.abs(compSpectra)
        f0 = float("-inf")
        for x in range(len(magnitude)):
            if magnitude[x] > f0:
                f0 = magnitude[x]
        f0List.append(f0)

    vals = np.array(f0List)

    return f0List

def autocorrelate(name):

    # Part 2

    input = Signal()
    input.wav_read(name)
    data = input.data

    size = int(len(data) / 2048)
    chunk = 2048
    currChunk = [0.0] * chunk
    f0List = []

    while chunk < 2048 * size:
        counter = 0
        for x in range(chunk - 2048, chunk):
            currChunk[counter] = data[x]
            counter += 1
        chunk += 2048

        # Autocorrelation
        corrDFT = fft(currChunk)
        for x in range(len(corrDFT)):
            corrDFT[x] = corrDFT[x] * np.conjugate(corrDFT[x])
        corrIDFT = ifft(corrDFT)

        magnitude = np.abs(corrIDFT)

        f0 = float("-inf")
        for x in range(len(magnitude)):
            if magnitude[x] > f0:
                f0 = magnitude[x]
        f0List.append(f0)

    vals = np.array(f0List)

    return f0List

def summationf0AC(name):

    #sumtest does same thing in less code, delete this

    input = Signal()
    input.wav_read(name)
    data = input.data

    size = int(len(data) / 2048)
    chunk = 2048
    currChunk = [0.0] * chunk
    f0List = []

    while chunk < 2048 * size:
        counter = 0
        for x in range(chunk - 2048, chunk):
            currChunk[counter] = data[x]
            counter += 1
        chunk += 2048

        cc1 = currChunk
        cc2 = currChunk

        # autocorrelation
        corrDFT = fft(cc1)
        for x in range(len(corrDFT)):
            corrDFT[x] = corrDFT[x] * np.conjugate(corrDFT[x])
        corrIDFT = ifft(corrDFT)
        magnitude1 = np.abs(corrIDFT)
        x1 = float("-inf")
        for x in range(len(magnitude1)):
            if magnitude1[x] > x1:
                x1 = magnitude1[x]

        #f0 maximum
        compSpectra = fft(currChunk)
        magnitude2 = np.abs(compSpectra)
        x2 = float("-inf")
        for x in range(len(magnitude2)):
            if magnitude2[x] > x2:
                x2 = magnitude2[x]

        x0 = x1 + x2
        f0List.append(x0)
    vals = np.array(f0List)
    return f0List

def sumtest(name):

    val1 = f0max(name)
    val2 = autocorrelate(name)
    val = [0.0] * len(val2)

    for x in range(len(val1)):
        val[x] = val1[x] + val2[x]

    return val

def main():

    filename = 'qbh1.wav'

    one = f0max(filename)
    one = np.array(one)

    two = autocorrelate(filename)
    two = np.array(two)

    plt.plot(one, 'r', label='fundamental max')
    plt.plot(two, 'b', label='autocorrelate')

    three = summationf0AC(filename)
    three = np.array(three)

    plt.plot(three, 'g', label='summation')
    plt.legend()
    plt.title('QBH')
    plt.show()

if __name__ == "__main__":
    main()