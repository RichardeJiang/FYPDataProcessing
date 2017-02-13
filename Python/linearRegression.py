import os
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def writeScore(dictFile, fileName):
	theFile = open(fileName, "w")
	for item in dictFile:
		theFile.write("%s: " % str(item))
		theFile.write("%s\n" % str(dictFile[item]))
	theFile.close()
	return

def readTimeSeriesData(fileName):
	fp = open(fileName, "r")
	phraseList = []
	timeSeries = []
	for line in fp:
		temp = line.split(":")
		phraseList.append(temp[0])
		timeSeries.append([float(ele) for ele in temp[1].split(" ")])

	return phraseList, timeSeries

def splitData(timeSeries):
	resultXList = []
	resultYList = []
	for phraseSeries in timeSeries:
		resultX = []
		resultY = []
		phraseSeries = phraseSeries[:-4]
		#print len(phraseSeries)
		size = len(phraseSeries)
		for i in range(size - 3):
			resultX.append(phraseSeries[i: i+3])
			resultY.append([phraseSeries[i+3]])

		resultXList.append(resultX)
		resultYList.append(resultY)
	return resultXList, resultYList

if (__name__ == "__main__"):
	fileName = "keyPhraseTimeSeries.txt"
	phraseList, timeSeries = readTimeSeriesData(fileName)

	dataXList, dataYList = splitData(timeSeries)
	dataXList = np.asarray(dataXList)
	dataYList = np.asarray(dataYList)

	regression = linear_model.LinearRegression()
	ridge = linear_model.Ridge(alpha = 0.5)
	phraseCount = len(phraseList)
	meanSquareError = {}
	varianceScore = {}
	meanSquareErrorRidge = {}
	varianceScoreRidge = {}
	for index in range(phraseCount):
		XList = dataXList[index]
		YList = dataYList[index]
		XTrain, XTest, YTrain, YTest = train_test_split(XList, YList, test_size = 0.1, random_state = 42)
		regression.fit(XTrain, YTrain)
		ridge.fit(XTrain, YTrain)

		meanSquareError[phraseList[index]] = np.mean((regression.predict(XTest) - YTest) ** 2)
		varianceScore[phraseList[index]] = regression.score(XTest, YTest)

		meanSquareErrorRidge[phraseList[index]] = np.mean((ridge.predict(XTest) - YTest) ** 2)
		varianceScoreRidge[phraseList[index]] = ridge.score(XTest, YTest)

		plt.scatter(regression.predict(XTrain), regression.predict(XTrain) - YTrain, color = 'b', s =40, alpha = 0.5)
		plt.scatter(regression.predict(XTest), regression.predict(XTest) - YTest, color = 'g', s = 40)
		plt.hlines(y = 0, xmin = 0, xmax = 0.1)
		plt.title("Train: blue; Test: green")
		plt.ylabel("Residuals")
		plt.savefig("plots90/" + str(index + 1) + "-" + str(phraseList[index]) + "-linear.png")
		plt.close()

		plt.scatter(ridge.predict(XTrain), ridge.predict(XTrain) - YTrain, color = 'b', s =40, alpha = 0.5)
		plt.scatter(ridge.predict(XTest), ridge.predict(XTest) - YTest, color = 'g', s = 40)
		plt.hlines(y = 0, xmin = 0, xmax = 0.00000001)
		plt.title("Train: blue; Test: green")
		plt.ylabel("Residuals")
		plt.savefig("plots90/" + str(index + 1) + "-" + str(phraseList[index]) + "-ridge.png")
		plt.close()

	writeScore(meanSquareError, "data90/meanLinear.txt")
	writeScore(varianceScore, "data90/varianceLinear.txt")

	writeScore(meanSquareErrorRidge, "data90/meanRidge.txt")
	writeScore(varianceScoreRidge, "data90/varianceRidge.txt")
	pass