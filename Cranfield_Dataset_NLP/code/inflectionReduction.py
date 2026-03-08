from util import *

# Add your import statements here
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

class InflectionReduction:
	## in this class we will define function that will help in reducing the inflection and bringing down any given word
	## to its root. 
	def porterStemmer(self, text):
		## this defines the stemming technique, where in the word like running becomes run and studying becomes studi.
		## this is not an efficient way of inflection, but works fast and fine with many applications. 
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
		## in this function we define the lemmatization technique based on the WordNet database. 
		## lemmatization essentially reduces the word to its base form which is always a valid dictionary word. 
		## for example, the word "better" will be reduced to "good" and "running" will be reduced to "run".
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
		## helper function for porter stemmer's method.
		reducedText = self.porterStemmer(text)

		return reducedText