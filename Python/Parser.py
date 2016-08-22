from xml.dom.minidom import parse
import xml.dom.minidom
import re

class XmlParser:
	''' The general-purpose Xml parser '''

	def __init__(self, fileList, rootTag, targetTags):
		self.fileList = fileList
		self.rootTag = rootTag
		self.targetTags = targetTags
		self.content = ""

	def parse(self):

		count = 0
		result = ""

		for fileName in self.fileList:
			DOMTree = xml.dom.minidom.parse(fileName)
			collection = DOMTree.documentElement

			#articles = collection.getElementsByTagName("article_rec")
			articles = collection.getElementsByTagName(self.rootTag)
			regexBracket = re.compile(r'<.*?>', re.IGNORECASE)
			regexQuote = re.compile(r'\"', re.IGNORECASE)

			for article in articles:

				for tag in self.targetTags:
					tagContents = article.getElementsByTagName(tag)
					#print tagContent

					if (tag != "article_publication_date"):
						for tagContent in tagContents:
							tagContent = re.sub(r'<.*?>', "", tagContent.childNodes[0].data)
							tagContent = re.sub(r'\"', "", tagContent)
							# tagContent = regexBracket.sub("", tagContent)
							# tagContent = regexQuote.sub("", tagContent)
							self.content += tagContent
							self.content += " "

					else:
						for time in tagContents:
							timeList = time.childNodes[0].data.split("-")
							timing = timeList[len(timeList) - 1]
							self.content = timing + " " + self.content
				# titles = article.getElementsByTagName("title")
				# abstracts = article.getElementsByTagName("par")
				# timeStamp = article.getElementsByTagName("article_publication_date")

				# for title in titles:
				# 	title = regexBracket.sub("", title)
				# 	title = regexQuote.sub("", title)
				# 	self.content += title
				# 	self.content += " "

				# for abstract in abstracts:
				# 	abstract = regexBracket.sub("", abstract)
				# 	abstract = regexQuote.sub("", abstract)
				# 	self.content += abstract
				# 	self.content += " "

				# for time in timeStamp:
				# 	timeList = time.split('-')
				# 	timing = timeList[len(timeList) - 1]
				# 	self.content = timing + " " + self.content
				if (len(self.content.split(" ")) <= 20):
					continue

				if self.content:
					self.content = self.content.strip()
					result += (str(count) + " en " + self.content + "\n")
					self.content = ""
					count += 1

		return result