import numpy as np
import matplotlib.pyplot as plt
import os

def readFile(filePath):
	fp = open(filePath, 'r')
	lines = fp.readlines()
	matrix = []
	for line in lines:
		lineContent = line.split(',')
		matrix.append(lineContent)
	return matrix

def plotMatrix(folderPath, matrix, index):
	size = len(matrix[0]) - 2
	X = range(0, size)
	for line in matrix:
		temp = line[2 : size+2]
		Y = [float(i) for i in temp]
		#plt.plot(X, Y, label=line[0])
		plt.plot(X, Y, label=str(line[0]))
		plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0., fontsize='small')
		#plt.savefig(folderPath + "/" + str(index) + "-top.png")
	#fig = plt.figure()
	plt.savefig(folderPath + "/" + str(index) + "-top.png")
	#plt.show()
	plt.close()
	
	return

if (__name__=="__main__"):
	dirList = []
	for dirName in os.listdir('.'):
		if (dirName.endswith('0')):
			dirList.append(dirName)

	for dirName in dirList:
		for index in range(0,8):
			path = dirName + '/test' + str(index) + 'top20words.dat'
			matrix = readFile(path)
			plotMatrix(dirName, matrix, index)