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


# Assignment 3

## Question 1

Download the 1.2 GB genre classification dataset from:
http://marsyas.info/downloads/datasets.html

You will only need 1.2 GB of space for download but after that you can pick
any three genres out of the 10 genres for your experiments. Alternatively if
you don’t have enough space you can download individual files for 3 genres
(at least 20 tracks for each genre) from:
http://marsyas.cs.uvic.ca/sound/genres/

Read the instructions in Chapter 3 of the Marsyas User Manual (Tour -
Command Line Tools) and use the bextract command-line program to
extract features for the 3 genres you selected. Load the extracted .arff
file into Weka and report on the classification accuracy of the following
classifiers: ZeroR, NaiveBayesSimple, J48, and SMO.

<b>Part 1:</b> Your deliverable will be the list of command you used and the classification
accuracy + confusion matrix for each classifier for the 3-genre
experiment.

<b>Part 2:</b> Now use Weka to convert the .arff to the .libsvm format that is supported
by scikit-learn. Do a similar experiment using scikit-learn i.e 3 classifiers and
report accuracy and confusion matrix. Provide a listing of the relevant code

## Question 2

Text categorization is the task of assigning a given document to one of a fixed
set of categories, on the basis of text it contains. Naive Bayes models are
often used for this task. In these models, the query variable is the document
category, and the “effect” variables are the presence/absence of each word
in the language; the assumption is that words occur independently in documents
within a given category (conditional independence), with frequencies
determined by document category.

Our goal will be to build a simple Naive Bayes classifier for the MSD
dataset, which uses lyrics to classify music into genres. More complicated
approaches using term frequency and inverse document frequency weighting
and many more words are possible but the basic concepts are the same.
The goal is to understand the whole process, so do not use existing machine
learning packages but rather build the classifier from “scratch”.

We are going to use the musicXmatch 1 dataset which is a large collection
of song lyrics in bag-of-words format for some of the trak IDS contained in
the Million Song dataset (MSD). The correspondent genre annotations, for
some of the song in the musicXmatch dataset, is provided by the MSD Allmusic
Genre Dataset 2. For the purpose of this course, in order to simplify
the problem, we are going to use a reduced version of the musicXmatch
dataset. Three genres are considered, namely: “Rap”, “Pop Rock”, and
“Country”. The resulting genre annotated dataset is obtained by an intersection
of musicXmatch and MAGD, where we select 1000 instances of
each genre, such that the three classes are balanced and easy to handle. In
addition, we also reduce the cardinality of the dictionary of words used for
the bag-of-words lyrics representation (originally equal to 5000), to the 10
best words for each genre. Intuitively, the best words are the most frequent
words for a particular genre that are not frequent among all the genres 3
.
The resulting dictionary of the three genres is:

[ ’ de ’ , ’ ni g g az ’ , ’ ya ’ , ’ und ’ , ’ y all ’ , # rap<br>
’ ich ’ , ’ fuck ’ , ’ s hi t ’ , ’ yo ’ , ’ bi t c h ’ , # rap<br>
’ end ’ , ’ wait ’ , ’ again ’ , ’ l i g h t ’ , ’ eye ’ , # rock<br>
’ noth ’ , ’ l i e ’ , ’ f a l l ’ , ’ our ’ , ’ away ’ , # rock<br>
’ gone ’ , ’ good ’ , ’ ni gh t ’ , ’ blue ’ , ’ home ’ , # country<br>
’ long ’ , ’ l i t t l ’ , ’ w ell ’ , ’ he a r t ’ , ’ old ’ ] # country

An additional simplification of the problem is to consider just the presence
or absence of a particular word, instead of the frequency count. Therefore according
to this problem setup, the feature vector of the song TRAAAHZ128E0799171
is [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1,
0]

For answering this question we provide you with:

• data.npz – the three genres dataset (not binarized)

• labels.npz – the genre labels where Rap=12, Pop Rock=1, and Country=3)

• dictionary.pck – the full 5000 words dictionary

• words.pck – the 30 best word indexes with respect to the full dictionary

• tracks.pck – the track IDs of songs used.

. These data is available in either Python pickle format (*.pck), or NumPy
format (*.npz) and can be found at: http://marsyas.cs.uvic.ca/csc475_
asn3_data.tar.gz

<b>Part 1:</b> Write code that calculates the probabilities for each dictionary
word given the genre. For the purposes of this assignment we are
considering only the tracks belonging to the three genres: Rap, Rock
pop, Country

<b>Part 2:</b> Explain how these probability estimates can be combined to form
a Naive Bayes classifier. Calculate the classification accuracy and confusion
matrix that you would obtain using the whole data set for both
training and testing partitions.

<b>Part 3:</b> Read the Wikipedia page about cross-validation in statistics 4
.
Calculate the classification accuracy and confusion matrix using the
k−fold cross-validation, where k = 10. Note that you would use both
the training and testing data and generate your own splits.

<b>Part 4:</b> One can consider the Naive Bayes classifier a generative model
that can generate binary feature vectors using the associated probabilities
from the training data. The idea is similar to how we do direct
sampling in Bayesian Networks and depends on generating random
number from a discrete distribution (the unifying underlying theme
of this assignment). Describe how you would generate random genre
“lyrics” consisting solely of the words from the dictionary using your
model. Show 5 examples of randomly generated tracks for each of
the three genres: Rap, Rock pop, and Country; each example should
consist of a subset of the words in the dictionary.


# Assignment 4

## Question 1

Check out the tangible music interfaces in http://modin.yuri.at/tangibles.
Look at all the categories(tangibles, blocks, toys). Pick 3 that you find interesting
and write a short summary (1 paragraph for each interface) of what
you find interesting about them.

## Question 2

Propose a tangible interface that is somehow combined or related to the concepts
we have covered in this course. Basically it should somehow combine
the algorithms/tasks we have learned with some form of physical tangible
interaction. Describe the user interaction and motivate its usage contrasting
it with a traditional screen/keyboard/mouse graphical user interface. Although
you don’t need to do any hardware design try to propose something
that can be engineered using existing technologies.


# Assignment 5

## Question 1

Vamp is an audio processing plugin system for plugins that extract descriptive
information from audio data typically referred to as audio analysis
plugins or audio feature extraction plugins. It supports multiple types
of outputs and output types and there are a variety of tools for using
(and making VAMP plugins, something we won’t do). More information
about VAMP plugins, the associated tools, and design can be found at:
http://www.vamp-plugins.org/.

The goal of this question is to apply structure segmentation using two
plugins: the Segmentino plugin, and the Segmenter from the Queen Mary
plugin set. Both plugins can be obtained from: http://www.vamp-plugins.
org/download.html.

<b>Part 1:</b> Using the Sonic Visualizer tools, visulaize the segmentation results of
the two plugins for two audio files of your choice. Provide screenshots
of the segmentation results

<b>Part 2:</b> Using the Sonic Annotator tool, show how you can process the same
two files from the commend line. Show the commands you used 

<b>Part 3:</b> Using any tool do a structural segmentation of the audio yourself.
For example you can use Audacity and a label track to annotate the
segments. Convert your manual segentation and the two automatic
segmentations for each song to the same format and compare them by
listening. Describe in words what are the differences 

## Question 2

MIREX is the annual Music Information Retrieval Evaluation Exchange.
You can read about it at: http://www.music-ir.org/mirex/wiki/MIREX_
HOME. mir eval is a great Python package that implements many of the evaluation
metrics for the various MIR tasks. The objective of this question is
to use mir eval to perform a more formal computer evaluation of the segmentation
results from the previous question. You can read about the segmentation
evaluation metrics at: https://craffel.github.io/mir_eval/
#module-mir_eval.segment

<b>Part 1:</b> Use the vampy host to run the segmentation plugins from the previous
question inside Python for the two files you processed. You can
find the code at: https://code.soundsoftware.ac.uk/projects/
vampy-host.

<b>Part 2:</b> Connect the vampy code for computing the segmentation with the
mir eval code for evaluation. Print the pair-wise F-measure for all
4 combinations of track and segmentation plugin using your manual
structure segmentation as the ground truth. Discuss whether the pairwise
F-measure corresponds to your listening perception of segmentation
quality
