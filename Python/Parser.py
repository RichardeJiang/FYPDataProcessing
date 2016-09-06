from xml.dom.minidom import parse
import xml.dom.minidom
import re
import nltk

class XmlParser:
	''' The general-purpose Xml parser '''

	def __init__(self, fileList, rootTag, targetTags):
		self.fileList = fileList
		self.rootTag = rootTag
		self.targetTags = targetTags
		self.content = ""

	def tagNPFilter(self, sentence):
		tokens = nltk.word_tokenize(sentence)
		tagged = nltk.pos_tag(tokens)
		NPgrammar = r"""NP:{<DT>?<JJ|NN|NNS>*<NN|NNS>}
		ND:{<DT>?<NN|NNS><IN><DT>?<JJ|NN|NNS>*}"""   # pattern "a beautiful picture"
		#possible development: pattern "a girl in red shirt"
		#NPgrammar += "{<DT>?<NN|NNS><IN><DT>?<JJ|NN|NNS>*}" commented out for the fpllowing reasons
		#Problem: "a powerful computer with strong support from university" 
		#1, nested; 2, 'computer' is the keywords? or 'computer with support' is the keywords?
		cp = nltk.RegexpParser(NPgrammar)
		resultTree = cp.parse(tagged)   #result is of type nltk.tree.Tree
		result = ""
		for node in resultTree:
			if (type(node) == nltk.tree.Tree):
				#result += ''.join(item[0] for item in node.leaves()) #connect every words
				#result += node.leaves()[len(node.leaves()) - 1][0] #use just the last NN
				if (node.label() == 'NP'):   # NN phrases
					result += node.leaves()[len(node.leaves()) - 1][0]
				else:    # IN phrases
					if (node[0][1] == 'NN' or node[0][1] == 'NNS'):    # the first element is NN
						result += node[0][0]
					else:    # the first element is DT
						result += node[1][0]
					#result += node.leaves()[0][0]
			else:
				result += node[0]
			result += " "
		return result

	def keyWordFilter(self, article, keyword, filterTags):
		# the idea is to use citation, reference, keyword list to find software engineering related articles
		for filterTag in filterTags:
			tagContents = article.getElementsByTagName(filterTag)

			for tagContent in tagContents:
				if (keyword in tagContent.childNodes[0].data.lower()):
					return True
		
		return False

	def parse(self):

		count = 1
		result = ""

		keyword = "software engineering"
		filterList = ["kw", "ref_text", "cited_by_text", "concept_desc", "subtitle"]

		for fileName in self.fileList:
			try:
				DOMTree = xml.dom.minidom.parse(fileName)
			except xml.parsers.expat.ExpatError, e:
				print "The file causing the error is: ", fileName
				print "The detailed error is: %s" %e
			collection = DOMTree.documentElement

			#articles = collection.getElementsByTagName("article_rec")
			articles = collection.getElementsByTagName(self.rootTag)
			regexBracket = re.compile(r'<.*?>', re.IGNORECASE)
			regexQuote = re.compile(r'\"', re.IGNORECASE)

			for article in articles:

				if not self.keyWordFilter(article, keyword, filterList):
					pass
				else:

					for tag in self.targetTags:
						tagContents = article.getElementsByTagName(tag)
						#print tagContent

						if (tag != "article_publication_date"):
							for tagContent in tagContents:
								tagContent = re.sub(r'<.*?>', "", tagContent.childNodes[0].data)
								tagContent = re.sub(r'\"', "", tagContent)
								# tagContent = regexBracket.sub("", tagContent)
								# tagContent = regexQuote.sub("", tagContent)
								tagContent = self.tagNPFilter(tagContent)
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