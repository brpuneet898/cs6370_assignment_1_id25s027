from util import *

# Add your import statements here
from nltk.corpus import stopwords

class StopwordRemoval():

	def fromList(self, text):
		stop_words = set(stopwords.words("english"))

		stopwordRemovedText = []

		for sentence in text:
			filtered_sentence = []

			for word in sentence:
				if word.lower() not in stop_words:
					filtered_sentence.append(word)

			stopwordRemovedText.append(filtered_sentence)

		return stopwordRemovedText