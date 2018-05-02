import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#stopwords array
stopwords = ['la','de','el','podrias','por','favor']

if __name__ == '__main__':
	sentence= 'Podiras reproducirla cancion de el noa noa , por favor !'

	#tokenizar
	tokens = sentence.split(' ')

	#sentence cleaning

	clean_token=[]

	for token in tokens:

		#remove punctuation
		if all(char in set(string.punctuation) for char in token):
			continue

		#remove numbers
		if token.isdigit():
			continue

		#transform the token to lowcase and remove sentences
		token= token.lower()
		token = token.strip()

		#remove stopworld
		if token in stopwords:
			continue

		

		clean_token.append(token)

	#print(tokens)
	#print(clean_token)

	#solo en la procatica
	temp=[]
	temp.append(' '.join(clean_token))


	#bag of words transformation
	conunt_vect = CountVectorizer()
	bag_of_words_array = conunt_vect.fit_transform(temp)

	print(bag_of_words_array)
	print(conunt_vect.vocabulary_)

	#model training
	naive_bayes_classifier = MultinomialNB()
	naive_bayes_classifier.fit(bag_of_words_array,['1'])

	print(naive_bayes_classifier.predict(bag_of_words_array))


####SUBIR EL ARCHIVO A GIT########
#modificar para que aprenda 3 oraciones
#creando un arreglo de oraciones
#para el predict poner un string