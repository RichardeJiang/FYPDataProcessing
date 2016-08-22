import os
from xml.dom.minidom import parse
import xml.dom.minidom
from Parser import XmlParser

def writeToFile(filePath, content):
	fp = open(filePath, 'w')
	#content.encode('ascii', 'ignore')
	fp.write(content.encode('ascii', 'ignore'))
	fp.close()
	return

def parseXml(fileList):
	rootTag = "article_rec"
	targetTags = ["title", "par", "article_publication_date"]
	parser = XmlParser(fileList, rootTag, targetTags)
	return parser.parse()

if (__name__=='__main__'):
	dirList = os.listdir('.')
	fileList = []
	for fileName in dirList:
		if fileName.endswith('.xml'):
			fileList.append(fileName)

	procTitle = fileList[0].split('-')[0]
	areaTitle = fileList[0].split('-')[1]
	wholeList = []
	procList = []
	areaList = []
	for name in fileList:
		contents = name.split('-')
		proc = contents[0]
		area = contents[1]
		if (procTitle == proc):
			if (areaTitle == area):
				#areaList.append(name)
				pass
			else:
				procList.append(areaList)
				areaList = []
				areaTitle = area
				#areaList.append(name)
			areaList.append(name)
		else:
			procList.append(areaList)
			wholeList.append(procList)
			procList = []
			areaList = []
			areaTitle = area
			procTitle = proc

	count = 0
	for procs in wholeList:
		#print item
		for areas in procs:
			parsedValues = ""
			#doc = "data-" + areas[0].split('-')[0] + "-" + areas[0].split('-')[1] + ".txt"
			#for fileName in areas:
			doc = "testPy/data-" + str(count) + ".txt"
			count += 1
			parsedValues = parseXml(areas)
			writeToFile(doc, parsedValues)
	pass