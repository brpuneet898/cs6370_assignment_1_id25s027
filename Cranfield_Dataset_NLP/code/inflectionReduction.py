from util import *

# Add your import statements here
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

class InflectionReduction:

	def porterStemmer(self, text):
		ps = PorterStemmer()
		reducedText = []

		for sentence in text:
			stemmed_sentence = []
			for word in sentence:
				stemmed_word = ps.stem(word)
				stemmed_sentence.append(stemmed_word)

			reducedText.append(stemmed_sentence)

		return reducedText

	def wordnetLemmatizer(self, text):
		lemmatizer = WordNetLemmatizer()
		reducedText = []

		for sentence in text:
			lemmatized_sentence = []
			for word in sentence:
				lemma = lemmatizer.lemmatize(word)
				lemmatized_sentence.append(lemma)

			reducedText.append(lemmatized_sentence)

		return reducedText

	def reduce(self, text):
		reducedText = self.porterStemmer(text)

		return reducedText