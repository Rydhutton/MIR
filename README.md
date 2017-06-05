# MIR
Music Information Retrieval (CSC475 - University of Victoria - Winter 2017)

# Assignment 1

## Question 1

Using a programming language of your choice write code to directly compute the Discrete Fourier Transform (DFT) of an input array. You should express everything directly at low level using array access, for loops, and arithmetic operations i.e do not use a complex number type if the language supports it, and do not use any vector/matrix multiplication facilities.

<b>Part 1:</b> Provide a listing of your code and a plot showing that your algorithm produces the same magnitude response as a Fast Fourier Transform routine in your language of choice that is either built in or freely available.
<b>Part 2:</b> For testing use a linear combination of 3 sinusoids that are spaced harmonically and have different amplitudes and phases. Test a fundamental frequency that corresponds to a particular DFT bin as well as a fundamental frequency that is between DFT bins. Plot the correpsonding magnitude and phase spectrums and comment on what you observe.
<b>Part 3:</b> Plot the basis functions i.e the sine and cosine signals for a particular DFT bin (let’s say bin 4 in a 256 point DFT)
<b>Part 4:</b> Plot the result of point-wise multiplying your input signal from the previous subquestion with the three harmonics with the two basis functions that correspond to the closest DFT bin. Also plot the result of point-wise multiplying your input signal with two basis functions that correspond to an unrelated DFT bin. What do you observe ? How do these plots connect with your understanding of the magnitude and phase for a DFT bin
