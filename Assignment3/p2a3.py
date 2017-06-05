import csv
import random
import math
import numpy as np
from sklearn.naive_bayes import GaussianNB

def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def numpyToCSV(array):
    np.savetxt("test.csv", array, delimiter=",")

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    if(len(numbers) - 1 <= 0):
        variance = 0
    else:
        variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbability(x, mean, stdev):
    if(stdev <= 0):
        stdev = 20000
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0




def main():

    dictionary = np.load("dictionary.pck")
    tracks = np.load("tracks.pck")
    labelsGARBAGE = np.load("labels.npz")
    wordsGARBAGE = np.load("words.npz")
    dataGARBAGE = np.load("data.npz")
    labels = labelsGARBAGE['arr_0']
    wordsNums = wordsGARBAGE['arr_0']
    data = dataGARBAGE['arr_0']

    """
    words = ['de', 'niggaz', 'ya', 'und', 'yall', 'ich', 'fuck', 'shit', 'yo', 'bitch', #rap
    'end', 'wait', 'again', 'light', 'eye', 'noth', 'lie', 'fall', 'our', 'away',		#rock
    'gone', 'good', 'night', 'blue', 'home', 'long', 'littl', 'well', 'heart', 'old']	#country
    """

    #note in labels Rap=12, Pop_Rock=1 and Conutry=3

    words = [""] * 30

    for i in range(30):
        x = wordsNums[i]
        words[i] = dictionary[x]


    # PART 1 ----------------------------------------------------------

    # Write code that calculates the probabilities for each dictionary
    # word given the genre. For the purposes of this assignment we are
    # considering only the tracks belonging to the three genres: Rap, Rock
    # pop, Country (1pt, 0.5pt)

    rapProbs = [0.0] * 10
    popProbs = [0.0] * 10
    countryProbs = [0.0] * 10

    for i in range(1000):
        for j in range(10):
            rapProbs[j] += data[i][j]
    for i in range(1000, 2000):
        for j in range(10):
            popProbs[j] += data[i][j+10]
    for i in range(2000, 3000):
        for j in range(10):
            countryProbs[j] += data[i][j+20]

    rCount = sum(rapProbs)
    pCount = sum(popProbs)
    cCount = sum(countryProbs)

    # Normalizing
    for x in range(10):
        rapProbs[x] = rapProbs[x] / rCount
        popProbs[x] = popProbs[x] / pCount
        countryProbs[x] = countryProbs[x] / cCount

    # ToString
    for x in range(10):
        print(words[x] + ": \t" + str(rapProbs[x]))
    for x in range(10):
        print(words[x+10] + ":\t" + str(popProbs[x]))
    for x in range(10):
        print(words[x+20] + ":\t" + str(countryProbs[x]))


    # PART 2 -------------------------------------------------------------------

    # Explain how these probability estimates can be combined to form
    # a Naive Bayes classifier. Calculate the classification accuracy and confusion
    # matrix that you would obtain using the whole data set for both
    # training and testing partitions. (1pt, 0.5pt)

    splitRatio = 0.67

    #trainingSet, testSet = splitDataset(data, splitRatio)
    #summaries = summarizeByClass(trainingSet)
    #predictions = getPredictions(summaries, testSet)
    #accuracy = getAccuracy(testSet, predictions)
    #print('Accuracy: {0}%').format(accuracy)

    filename = 'test.csv'
    ds = loadCsv(filename)
    trainingSet, testSet = splitDataset(ds, splitRatio)
    #print('Split {0} rows into train={1} and test={2} rows').format(len(ds))
    summaries = summarizeByClass(trainingSet)
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}%').format(accuracy)

    probs = rapProbs + popProbs + countryProbs
    clf1 = GaussianNB()
    #print probs
    probs2 = []
    for x in range(3000):
        probs2.append(probs)
    #print probs2
    clf1.fit(data, probs2)
    print(clf1.score(data, probs2))

if __name__ == "__main__":
    main()