import numpy as np
import math
import os
#import matplotlib.pyplot as plt

def sortTopWords(inputList):
	inputList.sort(key=lambda line: int(line[1]))

def plotSaveTopWords(folderPath, topWordList, index):
	size = len(topWordList[0]) - 2
	X = range(0, size)
	for line in topWordList:
		Y = line[2 : size+2]
		plt.plot(X, Y, label=line[0])
		print("X is: " + str(X))
		print("Y is: " + str(Y))
		fig = plt.figure()
		plt.savefig(folderPath + "/" + str(index) + "-top.jpg")
	
	return

def matchWord(dictPath, count):
	fp = open(dictPath, "r")
	lines = fp.readlines()
	if (count >= len(lines)):
		return ""
	elif (len(lines[count].split(",")) >= 2):
		return lines[count].split(",")[1].rstrip()
	else:
		return ""

def calculateAve(listInput):
	return reduce(lambda x, y: x+y, listInput)/len(listInput)

def readFile(filePath):
	fileContent = np.loadtxt(filePath)
	return fileContent

def findNumOfSeq(filePath):
	fp = open(filePath, "r")
	lines = fp.readlines()
	words = lines[2].split()
	return int(words[1])

def writeNewFile(filePath, newMatrix):
	fp = open(filePath, 'w')
	for line in newMatrix:
		fp.write("%s\n" % ','.join(str(part) for part in line))

if (__name__=="__main__"):
	dirList = []
	for dirName in os.listdir('.'):
		if(dirName.endswith('0')):
			dirList.append(dirName)

	for dirName in dirList:
		info_path = dirName + "/lda-seq/info.dat"
		numOfSeq = findNumOfSeq(info_path)
		for x in range(0, 8):
			topicName = dirName + "/test" + str(x) + ".dat"
			matrix = readFile(topicName).tolist()
			dictPath = dirName + "/dict.txt"
			#print matrix
			updatedMatrix = []
			count = 0
			for wordTopicDis in matrix:
				word = matchWord(dictPath, count)
				average = calculateAve(wordTopicDis)
				wordTopicDis.insert(0, word)
				wordTopicDis.insert(1, average)
				updatedMatrix.append(wordTopicDis)
				count += 1
			sortTopWords(updatedMatrix)
			top20Words = updatedMatrix[0:19]
			writeNewFile(dirName + "/test" + str(x) + "word.dat", updatedMatrix)
			writeNewFile(dirName + "/test" + str(x) + "top20words.dat", top20Words)
			#plotSaveTopWords(dirName, top20Words, x)
	#readFile()

	pass