
from nltk.corpus import stopwords
import numpy as np
import re

# se definen las stopwords para español
stop_words = stopwords.words('spanish')
stop_words = set(stop_words)

# abrimos el archivo con los documentos
with open('data.txt', 'r', encoding='utf-8') as data:
    document_words = data.read().lower()

# abrimos el archivo con las etiquetas
with open('labels.txt', 'r', encoding='utf-8') as data:
    labels = data.readlines()

# aquí se crea el vocabulario de los documentos y la frecuencia de cada término
d = {}
for word in re.findall(r'[a-zñáíúéó]+', document_words):
    if word not in stop_words and len(word) > 2 and len(word) < 20:
        d[word] = d.get(word, 0) + 1
vocabulary = [word for word in d]

# Aqui se crea la matriz de term-document-frecuency
tdfm = []
with open('data.txt', 'r', encoding='utf-8') as data:
    for line in data:
        line = line.lower()
        tokens = re.findall(r'[a-zñáíúéó]+', line)
        vector = []
        for word in vocabulary:
            vector.append(tokens.count(word))
        tdfm.append(vector)
tdfm = np.array(tdfm)


print(tdfm)
