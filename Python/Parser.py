from xml.dom.minidom import parse
import xml.dom.minidom
import re
import nltk
from nltk.stem.snowball import SnowballStemmer

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
		# NPgrammar = r"""NP:{<DT>?<JJ|NN|NNS>*<NN|NNS>}
		# ND:{<DT>?<NN|NNS><IN><DT>?<JJ|NN|NNS>*}"""
		NPgrammar = "NP:{<DT>?<JJ|NN|NNS>*<NN|NNS>}"
		#Problem: "a powerful computer with strong support from university" 
		#1, nested; 2, 'computer' is the keywords? or 'computer with support' is the keywords?
		cp = nltk.RegexpParser(NPgrammar)
		resultTree = cp.parse(tagged)   #result is of type nltk.tree.Tree
		result = ""
		stemmer = SnowballStemmer("english")
		for node in resultTree:
			if (type(node) == nltk.tree.Tree):
				#result += ''.join(item[0] for item in node.leaves()) #connect every words
				#result += stemmer.stem(node.leaves()[len(node.leaves()) - 1][0]) #use just the last NN

				if node[0][1] == 'DT':
					node.remove(node[0])  #remove the determiners
				currNounPhrase = ''.join(stemmer.stem(item[0]) for item in node.leaves())
				result += currNounPhrase

				if len(node.leaves()) == 1:
					pass
				else:
					result += ' '
					result += currNounPhrase #double noun phrases to increase the weight

				### The following part assumes nested grammar can be supported ###
				### which turns out to be false, so use the previous selction instead ###
				# if (node.label() == 'NP'):   # NN phrases
				# 	result += node.leaves()[len(node.leaves()) - 1][0]
				# else:    # IN phrases
				# 	if (node[0][1] == 'NN' or node[0][1] == 'NNS'):    # the first element is NN
				# 		result += node[0][0]
				# 	else:    # the first element is DT
				# 		result += node[1][0]
				### End of wasted part ###

			else:
				result += stemmer.stem(node[0])
			result += " "
		return result

	def keyWordFilter(self, article, keywords, filterTags):
		# the idea is to use citation, reference, keyword list to find software engineering related articles
		for filterTag in filterTags:
			tagContents = article.getElementsByTagName(filterTag)

			for tagContent in tagContents:
				for keyword in keywords:
					keywordDash = '-'.join(keyword.split(' '))
					if (keyword in tagContent.childNodes[0].data.lower()) 
						or (keywordDash in tagContent.childNodes[0].data.lower()):
						return True
		
		return False

	def labelAllocator(self, article):
		labels = ['agile software-development', 'program-comprehension program-visualization',
			'autonomic self-managed software', 'requirements engineering',
			'computer-supported collaborative work', 'reengineering reverse-engineering',
			'component-based software-engineering', 'quality performance',
			'configuration management deployment', 'service-oriented architectures applications',
			'dependendability safety reliability', 'software-architecture design',
			'distributed web-based internet-scale', 'software-economics software-metrics',
			'empirical software-engineering', 'software-evolution',
			'end-user software-engineering', 'software-maintenance',
			'engineering secure software', 'software-policy software-ethics',
			'feature interaction generative programming', 'software-reuse',
			'human social aspects', 'software-specifications',
			'knowledge-based software-engineering', 'testing analysis',
			'mobile embedded real-time systems', 'theory formal-methods',
			'model-driven software-engineering', 'tools environments',
			'patterns frameworks', 'validation verification',
			'processes workflow']
		labelCheckList = [0] * len(labels)
		returnedLabels = []
		targetTags = ['par', 'title', 'subtitle', 'ft_body', 'concept_desc', 'kw']
		contentString = ''
		for tag in targetTags:
			tagContents = article.getElementsByTagName(tag)
			for tagContent in tagContents:
				contentString += tagContent.childNodes[0].data.lower()
				contentString += ' '
		for i in range(0, len(labels)):
			label = labels[i]
			tokens = label.split(' ')
			for token in tokens:
				origin = token
				spaceDuplicate = token.replace('-', ' ')
				if origin in contentString or spaceDuplicate in contentString:
					labelCheckList[i] += 1

			if labelCheckList[i] >= len(tokens): # Q: how to set this threshold
				returnedLabels.append(','.join(tokens))

		if len(returnedLabels) > 0:
			return ' '.join(returnedLabels)
		else:
			return 'none'

	def parse(self):

		count = 1
		result = ""

		keywords = ["software engineering", "software and its engineering"]
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

				if not self.keyWordFilter(article, keywords, filterList):
					pass
				else:
					# add in label filters to allocate the labels
					tags = self.labelAllocator(article)
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
						#result += (str(count) + " en " + self.content + "\n")
						result += (str(count) + "\t" + tags + "\t" + self.content + "\n")
						self.content = ""
						count += 1

		return result