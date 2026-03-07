# from util import *

import re
import nltk
import spacy
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')

class SentenceSegmentation():

	def __init__(self):
		self.nlp = spacy.load("en_core_web_sm")

	def naive(self, text):
		sentences = re.split(r'[.!?]+', text)
		segmentedText = [s.strip() for s in sentences if s.strip() != ""]
		return segmentedText

	def punkt(self, text):
		segmentedText = sent_tokenize(text)
		return segmentedText

	def spacySegmenter(self, text):
		doc = self.nlp(text)
		segmentedText = [sent.text.strip() for sent in doc.sents]
		return segmentedText