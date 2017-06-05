# MIR
Music Information Retrieval (CSC475 - University of Victoria - Winter 2017)


# Assignment 1

## Question 1

Using a programming language of your choice write code to directly compute the Discrete Fourier Transform (DFT) of an input array. You should express everything directly at low level using array access, for loops, and arithmetic operations i.e do not use a complex number type if the language supports it, and do not use any vector/matrix multiplication facilities.

<b>Part 1:</b> Provide a listing of your code and a plot showing that your algorithm produces the same magnitude response as a Fast Fourier Transform routine in your language of choice that is either built in or freely available.

<b>Part 2:</b> For testing use a linear combination of 3 sinusoids that are spaced harmonically and have different amplitudes and phases. Test a fundamental frequency that corresponds to a particular DFT bin as well as a fundamental frequency that is between DFT bins. Plot the correpsonding magnitude and phase spectrums and comment on what you observe.

<b>Part 3:</b> Plot the basis functions i.e the sine and cosine signals for a particular DFT bin (let’s say bin 4 in a 256 point DFT)

<b>Part 4:</b> Plot the result of point-wise multiplying your input signal from the previous subquestion with the three harmonics with the two basis functions that correspond to the closest DFT bin. Also plot the result of point-wise multiplying your input signal with two basis functions that correspond to an unrelated DFT bin. What do you observe ? How do these plots connect with your understanding of the magnitude and phase for a DFT bin

## Question 2

<b>Part 1:</b> Using an existing FFT implementation, plot the DFT magnitude response
in dB (using a DFT size of 2048) of a sine wave that is exactly equal to the
center frequency of DFT bin 525. On the same plot overlap the DFT magnitude
response in dB of a sine wave with frequency that would correspond to
bin 600.5 (in a sense falling between the crack of two neigh- boring frequency
bins). Finally on the same plot overlap the DFT magnitude rsponse in dB
of the second sine wave (525.5) windowed by a hanning window of size 2048.
Write 2-3 brief sentences describing and explaning the three plots and what
is the effect of windowing to the magnitude response for these inputs. Make
sure your plots are in dB. 

<b>Part 2:</b> Find single note sound of an instrument (you can record yourself something
using Audacity, search Freesound, etc). Pick 2048 samples of audio
from the steady state portion of the sound (not the attack). Instruments
such a violin, flute, organ, saxophone that can sustain long notes are better
for this question. Repeat the process above with input your very short audio
segment from the steady state of the instrument tone. What is the effect of
windowing ? Show the corresponding plots.


# Assignment 2

The goal in this question is to extract a fundamental frequency contour from
an audio file. In order to test your code utilize the following audio input
signals:
• The little melody you created in assignment 1
• The same melody hummed by you and recorded in Audacity. Use a
single vowel like Ah for your singing.
• The qbhexamples.wav available under Resources in Connex.

## Question 1

A simple fundamental frequency estimation is to compute
the magnitude spectrum, select the highest peak, and return the
corresponding frequency as the result. Processing the sound in windows
will result in a time series of F0 estimates that can be plotted
over time. Make plots of the estimated FO over time for the three input
testing signals using a window of 2048 samples at 44100 sampling
rate.

<b>Part 1:</b> A simple fundamental frequency estimation is to compute
the magnitude spectrum, select the highest peak, and return the
corresponding frequency as the result. Processing the sound in windows
will result in a time series of F0 estimates that can be plotted
over time. Make plots of the estimated FO over time for the three input
testing signals using a window of 2048 samples at 44100 sampling
rate.

<b>Part 2:</b> An alternative approach is to compute the Autocorrelation of the signal.
Either use an existing implementation of Autocorrelation, or write
your own, or read about how it can be expressed using the DFT. The
peaks in the AutoCorrelation function correspond to different time domain
lags. Convert the lags to frequency and similarly to the previous
question plot the resulting F0 contours for the three input signals.

<b>Part 3:</b> Compare the F0 contours of the DFT approach and the
AutoCorrelation approach. Is one consistently better than the other ?
What types of errors do you observe ? Change your code so that both
the F0 estimates are computed for every window. Change your code
so that both the F0 estimates are computed for every window and plot
the sum of the two resulting F0 contours for each input signal.

## Question 2

The purpose of this question is to experiment with the audio feature called
spectral centroid. The centroid corresponds to a frequency and can easily be
computed from the magnitude spectrum. Similarly to the previous question
you will need process the audio in short chunks of 2048 samples and compute
the centroid for each chunk.

<b>Part 1:</b> Plot the centroid over time for the three examples that you used for monophonic
pitch estimation. In addition create audio files with a sine wave
generator that is controlled by the centroid to sonify the result. Listen
to the resulting audio and write some sentences about what you think is
happening.

<b>Part 2:</b> Download one or two classical music files and metal music files from the
corresponding folder of: http://marsyas.cs.uvic.ca/sound/genres/
Run two instances of your centroid sonification script at the same time
(using two command-line windows) to compare the results of pairs of different
input audio files:
• classical and classical
• classical and metal
• metal and metal
To some extent this would be what an automatic genre classification
system based purely on the Spectral Centroid would ”hear” to make a decision
of classical or metal. FYI using just the mean of the spectral centroid
a trained classifier makes the right decision 75% of the time. This is impressive
given that decisions are made every 20 milliseconds so even better
results could be obtained by some majority voting/filtering over the entire
file.
To see the influence of the texture window replaces each centroid value
with the average of the previous 20 centroid values and plot again the contours.
Then sonify the resulting smoother contour. Provide a very brief
commentary (no more than 3-4 sentences) of your observations on this soni-
fication experiment.
