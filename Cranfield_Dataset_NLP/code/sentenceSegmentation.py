from util import *

import re
import nltk
import spacy
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')

class SentenceSegmentation():

	def __init__(self):
		## initializing the spacy model and sentence segmenter class
		self.nlp = spacy.load("en_core_web_sm")

	def naive(self, text):
		## this is the basic top-down approach for sentence segmentation. 
		## here we are using regex to split the text into sentences based on the punctuation marks. 
		## we have used following special characters like - . ! ? as the main delimiters. 
		sentences = re.split(r'[.!?]+', text)
		## now if there are any empty sentences that might occur after splitting due to any reason, that will get removed now.
		segmentedText = [s.strip() for s in sentences if s.strip() != ""]
		return segmentedText

	def punkt(self, text):
		## here we are using the nltk library punkt sentence segmenter 
		segmentedText = sent_tokenize(text)
		return segmentedText

	def spacySegmenter(self, text):
		## we had already initialized the spacy model in the __init__ function.
		## we are going to use the same spacy model to segment the sentences.
		doc = self.nlp(text)
		segmentedText = [sent.text.strip() for sent in doc.sents]
		return segmentedText