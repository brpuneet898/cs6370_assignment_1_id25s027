import nltk
nltk.download('stopwords')
from util import *

# Add your import statements here
from nltk.corpus import stopwords


class StopwordRemoval():
	## in this class we will be defining some functions that will help us in removing the stopwords from the text.
	## what are stopwords? we can consider stopwords as those words which occur frequently in the corpus.
	## since they occur in most of the documents, then they do not posses much discriminative power and 
	## hence it is advised to remove them from the text .
	def fromList(self, text):
		## this function uses the standard ntlk stopword list of 198 words pre-defined in the english language.
		## this function go through the corpus and wherever it encounters a stopword, that will get removed. 
		stop_words = set(stopwords.words("english"))

		stopwordRemovedText = []

		for sentence in text:
			filtered_sentence = []

			for word in sentence:
				if word.lower() not in stop_words:
					filtered_sentence.append(word)

			stopwordRemovedText.append(filtered_sentence)

		return stopwordRemovedText