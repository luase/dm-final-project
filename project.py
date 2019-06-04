
from nltk.corpus import stopwords
import numpy as np
import re

# se definen las stopwords para español
stop_words = stopwords.words('spanish')
stop_words = set(stop_words)

# abrimos el archivo con los documentos
with open('data.txt', 'r', encoding='utf-8') as data:
    document_words = data.read().lower()

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

# Aqui creo el archivo nuevo con las n palabras más frecuentes de cada documento
n = 12
ind = range(len(vocabulary))
ind = np.array(ind)
document_keywords = []
for row in tdfm:
    # para cada renglón de la matriz de frecuencias
    mas_frecuentes = []
    for item in ind[np.argsort(row)[-n:]]:
        # encontrar las n palabras más repetidas
        mas_frecuentes.append(vocabulary[item])
    document_keywords.append(mas_frecuentes)
keywords = []
for keys in document_keywords:
    # generar los n/4 documentos nuevos con las palabras más frecuentes
    temp = int(n/4)
    for i in range(temp):
        mas_frecuentes = []
        mas_frecuentes.append(keys[-((i*2)+1)])
        mas_frecuentes.append(keys[-((i*2)+2)])
        mas_frecuentes.append(keys[(i*2)+1])
        mas_frecuentes.append(keys[(i*2)])
        keywords.append(mas_frecuentes)
# guardar los nuevos documentos
with open('keywords.txt', 'w', encoding='utf-8') as doc:
    for keys in keywords:
        for key in keys:
            doc.write("%s " % key)
        doc.write("\n")

# generar el archivo con las labels nuevas
# abrimos el archivo con las etiquetas
with open('labels.txt', 'r', encoding='utf-8') as data:
    labels = data.readlines()
with open('labels_kw.txt', 'w', encoding='utf-8') as doc:
    temp = int(n/4)
    for label in labels:
        for i in range(temp):
            doc.write("%s" % label)



# for row in tdfm:
#     print (row)
#     break
# print(d)