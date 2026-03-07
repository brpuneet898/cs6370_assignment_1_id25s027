from util import *

# Add your import statements here
import re
import nltk
import spacy
from nltk.tokenize import word_tokenize

class Tokenization():

	def __init__(self):
		self.nlp = spacy.load("en_core_web_sm")

	def naive(self, text):
		tokenizedText = []

		for sentence in text:
			tokens = re.findall(r'\b\w+\b', sentence)
			tokenizedText.append(tokens)

		return tokenizedText

	def pennTreeBank(self, text):
		tokenizedText = []

		for sentence in text:
			tokens = word_tokenize(sentence)
			tokenizedText.append(tokens)

		return tokenizedText

	def spacyTokenizer(self, text):
		tokenizedText = []

		for sentence in text:
			doc = self.nlp(sentence)
			tokens = [token.text for token in doc]
			tokenizedText.append(tokens)

		return tokenizedText