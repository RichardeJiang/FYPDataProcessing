import os
import numpy as np

def sortDict(inputList):
	inputList.sort(key=lambda line: int(line[0]))

def writeToNewDict(filePath, matrix):
	fp = open(filePath, 'w')
	#np.savetxt(filePath, matrix)
	for line in matrix:
		fp.write("%s" % ','.join(line))

def readDict(filePath):
	fp = open(filePath, 'r')
	lines = fp.readlines()
	contents = []
	for line in lines:
		content = line.split(',')
		contents.append(content)

	return contents

if (__name__=="__main__"):
	dictList = []
	fileList = os.listdir('dict')
	for fileName in fileList:
		if(fileName.endswith('dict.txt')):
			dictList.append('dict/' + fileName)

	for dictName in dictList:
		#matrix = np.loadtxt(dictName, delimiter=',').tolist()
		matrix = readDict(dictName)
		sortDict(matrix)
		filePath = dictName + '.new'
		writeToNewDict(filePath, matrix)

	pass