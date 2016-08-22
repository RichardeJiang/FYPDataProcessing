import os
from xml.dom.minidom import parse
import xml.dom.minidom

if (__name__='__main__'):
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

	for item in wholeList:
		print item
	pass