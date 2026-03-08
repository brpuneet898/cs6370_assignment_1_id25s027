from util import *

# Add your import statements here
import re
import nltk
import spacy
from nltk.tokenize import word_tokenize

class Tokenization():

	def __init__(self):
		## we are initializing the tokenizer class and also loading the spacy model . 
		self.nlp = spacy.load("en_core_web_sm")

	def naive(self, text):
		## this function implements a naive tokenization approach using regex.
		## we are using a simple rule that wherever its sees a word boundary or a you can say a whitespace
		## in simple terms on either side of the word, then that will be considered as 1 token. 
		## clearly this approach is bad because by this rule, running and run are two different tokens. 
		tokenizedText = []

		for sentence in text:
			tokens = re.findall(r'\b\w+\b', sentence)
			tokenizedText.append(tokens)

		return tokenizedText

	def pennTreeBank(self, text):
		## this tokenizer comes from the penn tree bank which is essenntially a corups. 
		## it also follows a top-down approach.
		## it has a set of linguistic tokenization rules derived from the above mentioned corpus. 
		## it defines how punctuations, abbreviations, and special characters have to be split. 
		tokenizedText = []

		for sentence in text:
			tokens = word_tokenize(sentence)
			tokenizedText.append(tokens)

		return tokenizedText

	def spacyTokenizer(self, text):
		## this function defines the spacy tokenizer on the model mentioned in the constructor.
		## it follows a hybrid approach like both top-down and bottom up. 
		## it has some rule-based tokenization which includes suffix, prefix, infix and exception rules.
		## and also uses statistical and lexical knowledge to tokenize the text.
		tokenizedText = []

		for sentence in text:
			doc = self.nlp(sentence)
			tokens = [token.text for token in doc]
			tokenizedText.append(tokens)

		return tokenizedText