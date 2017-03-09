import os
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter

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
	maxNum = 0.0
	for phraseSeries in timeSeries:
		resultX = []
		resultY = []
		phraseSeries = phraseSeries[:-4]
		if max(phraseSeries) > maxNum:
			maxNum = max(phraseSeries)
		size = len(phraseSeries)
		for i in range(size - 3):
			resultX.append(phraseSeries[i: i+3])
			resultY.append([phraseSeries[i+3]])

		resultXList.append(resultX)
		resultYList.append(resultY)

	print maxNum
	return resultXList, resultYList

def splitData2(timeSeries, windowSize):
	resultXList = []
	resultYList = []

	tempLine = timeSeries[0]
	size = len(tempLine) - windowSize
	seriesSize = len(timeSeries)
	for index in range(size):
		resultX = []
		resultY = []
		for lineIndex in range(seriesSize):
			resultX.append(timeSeries[lineIndex][index: index + windowSize])
			resultY.append(timeSeries[lineIndex][index + windowSize])
		resultXList.append(resultX)
		resultYList.append(resultY)

	return resultXList, resultYList, size

def formBinaryTrainingList(localCoef, XList, YList, globalCoefList):
	resultXList = []
	resultYList = []
	size = len(XList)
	years = globalCoefList.keys()
	startingYear = min(years)
	for index in range(size):
		x = XList[index]
		y = YList[index]
		globalCoef = globalCoefList[startingYear + index]
		print x
		print localCoef[0]
		print globalCoef
		localSum = np.dot(x, localCoef[0])
		globalSum = np.dot(x, globalCoef)
		resultYList.append(float(y - globalSum))
		resultXList.append([float(localSum - globalSum)])
	return resultXList, resultYList

def aggregatePhraseScore(keyPhraseScoreTimeSeries):
	aggregateCoefficient = 0.2
	years = len(keyPhraseScoreTimeSeries.values()[0])
	numOfPhrases = len(keyPhraseScoreTimeSeries)
	result = {key:[0 for n in range(years)] for key in keyPhraseScoreTimeSeries}
	for phrase, scoreList in keyPhraseScoreTimeSeries.iteritems():
		for index in range(0, years - 1):
			scoreList[index + 1] = scoreList[index + 1] * (1 - aggregateCoefficient) + scoreList[index] * aggregateCoefficient
		result[key] = scoreList
	return result

def aggregatePhraseScore(timeSeries, aggregateCoefficient):
	years = len(timeSeries[0])
	numOfPhrases = len(timeSeries)
	result = list(timeSeries)
	for phraseIndex in range(numOfPhrases):
		for index in range(years - 1):
			result[phraseIndex][index + 1] = result[phraseIndex][index + 1] * (1 - aggregateCoefficient) + result[phraseIndex][index] * aggregateCoefficient

	return result

def checkTopScoresInSeries(timeSeries, desiredCommonNumber = 3):
	npSeries = np.asarray(timeSeries)
	npSeries = np.transpose(npSeries)
	allowanceRange = 3
	# desiredCommonNumber = 3
	phraseIndexList = []
	for phraseScoreEachYear in npSeries:
		phraseApproximateScore = []
		identicalItems = []
		phraseIndexListYear = [[] for n in range(desiredCommonNumber)]
		for phraseScore in phraseScoreEachYear:
			approximateScore = int(phraseScore)
			# phraseApproximateScore.append(approximateScore)
			approximateSet = set(range(approximateScore - allowanceRange, approximateScore + allowanceRange + 1))
			# approximateSet = set([approximateScore - 2, approximateScore - 1, approximateScore, approximateScore + 1, approximateScore + 2])
			shownItems = list(set(identicalItems) & approximateSet)
			if len(set(shownItems)) == 0:
				identicalItems.append(approximateScore)
				phraseApproximateScore.append(approximateScore)
			else:
				phraseApproximateScore.append(min(shownItems))

		mostCommonTerms = [ele[0] for ele in Counter(phraseApproximateScore).most_common(desiredCommonNumber + 1)]
		if 0 in mostCommonTerms:
			mostCommonTerms = [ele in mostCommonTerms if ele != 0]
		else:
			mostCommonTerms = mostCommonTerms[:desiredCommonNumber]
		for index in range(len(phraseScoreEachYear)):
			# if phraseApproximateScore[index] == mostCommonTerms[0]
			if phraseApproximateScore[index] in mostCommonTerms:
				phraseIndexListYear[mostCommonTerms.index(phraseApproximateScore[index])].append(index) 
				# phraseIndexListYear.append(index)
		phraseIndexList.append(phraseIndexListYear)

	return phraseIndexList

if (__name__ == "__main__"):
	fileName = "phrase-agg30.txt"
	# fileName = "keyPhraseTimeSeries.txt"
	fileName = "keyPhraseTimeSeries5000WithScalingx100-archive2.txt"
	aggregateCoefficient = 0.3
	phraseList, timeSeries = readTimeSeriesData(fileName)
	#timeSeries = aggregatePhraseScore(timeSeries, aggregateCoefficient)
	windowSize = 3
	testSize = 20

	# This part is to compute the common scores to obtain the phrases sharing similar scores in one year
	commonPhrasesIndexList = checkTopScoresInSeries(timeSeries)
	
	# End of score computation part

	dataXList, dataYList = splitData(timeSeries)
	dataXList = np.asarray(dataXList)
	dataYList = np.asarray(dataYList)

	fullXList, fullYList, yearCover = splitData2(timeSeries, windowSize)
	fullXList = np.asarray(fullXList)
	fullYList = np.asarray(fullYList)

	regression = linear_model.LinearRegression()
	ridge = linear_model.Ridge(alpha = 0.5)
	phraseCount = len(phraseList)
	meanSquareError = {}
	varianceScore = {}
	coefRegression = {}
	meanSquareErrorRidge = {}
	varianceScoreRidge = {}

	for index in range(yearCover):
		XList = fullXList[index]
		YList = fullYList[index]
		print "XList length: " + str(len(XList))
		print "YList length: " + str(len(YList))

		XTrain, XTest, YTrain, YTest = train_test_split(XList, YList, test_size = float(testSize / 100.0), random_state = 42)
		print "XTrain:"
		print XTrain

		print "YTRain:"
		print YTrain
		regression.fit(XTrain, YTrain)

		meanSquareError[2015 - yearCover - windowSize + index] = np.mean((regression.predict(XTest) - YTest) ** 2)
		varianceScore[2015 - yearCover - windowSize + index] = regression.score(XTest, YTest)
		coefRegression[2015 - yearCover - windowSize + index] = regression.coef_

		# plt.scatter(regression.predict(XTrain), regression.predict(XTrain) - YTrain, color = 'b', s =40, alpha = 0.5)
		# plt.scatter(regression.predict(XTest), regression.predict(XTest) - YTest, color = 'g', s = 40)
		# plt.hlines(y = 0, xmin = 0, xmax = 100)
		# #plt.hlines(y = 0, xmin = 0, xmax = 0.1)
		# plt.title("Train: blue; Test: green; Starting Year: " + str(2015 - yearCover - windowSize + index))
		# plt.ylabel("Residuals")
		# # plt.savefig("full80/" + str(2015 - yearCover - windowSize + index) + "-" + str(windowSize) + "-linear-scaled.png")
		# plt.savefig("prehits/" + str(2015 - yearCover - windowSize + index) + "-" + str(windowSize) + "-agg-" + str(aggregateCoefficient) + ".png")
		# plt.close()


		plt.scatter(regression.predict(XTrain), YTrain - regression.predict(XTrain), color = 'b', s = 40, alpha = 0.5)
		plt.scatter(regression.predict(XTest), YTest - regression.predict(XTest), color = 'g', s = 40)
		plt.hlines(y = 0, xmin = 0, xmax = 100)
		plt.title("Train: blue; Test: green; Starting Year: " + str(2015 - yearCover - windowSize + index))
		plt.ylabel("Residuals")
		plt.savefig("posthits/" + str(2015 - yearCover - windowSize + index) + "-" + str(windowSize) + "-agg-" + str(aggregateCoefficient) + ".png")
		plt.close()

		break

	# writeScore(meanSquareError, "full80/meanLinear-" + str(100 - testSize) + "-" + str(windowSize) + "-scaled.txt")
	# writeScore(varianceScore, "full80/varianceLinear-" + str(100 - testSize) + "-" + str(windowSize) + "-scaled.txt")
	# writeScore(coefRegression, "full80/coef-" + str(100 - testSize) + "-" + str(windowSize) + "-scaled.txt")

	writeScore(meanSquareError, "posthits/meanLinear-" + str(100 - testSize) + "-" + str(windowSize) + ".txt")
	writeScore(varianceScore, "posthits/varianceLinear-" + str(100 - testSize) + "-" + str(windowSize) + ".txt")
	writeScore(coefRegression, "posthits/coef-" + str(100 - testSize) + "-" + str(windowSize) + ".txt")

	# coefRegressionPhrase = {}
	# alphaList = {}
	# for index in range(phraseCount):
	# 	XList = dataXList[index]
	# 	YList = dataYList[index]
	# 	XTrain, XTest, YTrain, YTest = train_test_split(XList, YList, test_size = float(testSize / 100.0), random_state = 42)
	# 	regression.fit(XTrain, YTrain)

	# 	phrase = phraseList[index]
	# 	localCoef = regression.coef_
	# 	coefRegressionPhrase[str(phraseList[index])] = localCoef

	# 	resultXList, resultYList = formBinaryTrainingList(localCoef, XList, YList, coefRegression)
	# 	XTrain, XTest, YTrain, YTest = train_test_split(resultXList, resultYList, test_size = float(testSize / 100.0), random_state = 42)
	# 	regression.fit(XTrain, YTrain)

	# 	alphaList[phrase] = regression.coef_

	# 	plt.scatter(regression.predict(XTrain), regression.predict(XTrain) - YTrain, color = 'b', s =40, alpha = 0.5)
	# 	plt.scatter(regression.predict(XTest), regression.predict(XTest) - YTest, color = 'g', s = 40)
	# 	plt.hlines(y = 0, xmin = 0, xmax = 100)
	# 	plt.title("Train: blue; Test: green; Phrase: " + phrase)
	# 	plt.ylabel("Residuals")
	# 	plt.savefig("mix" + str(windowSize) + "/" + str(index + 1) + "-" + str(phrase) + "-linear.png")
	# 	plt.close()

	# 	if index == 30:
	# 		break

	# writeScore(alphaList, "mix" + str(windowSize) + "/alphaList.txt")
	# pass