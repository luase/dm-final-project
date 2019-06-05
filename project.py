
from nltk.corpus import stopwords
from math import log
import numpy as np
import re

# se definen las stopwords para español
stop_words = stopwords.words('spanish')
stop_words = set(stop_words)
filename = 'data.txt'
filename2 = 'labels.txt'

# abrimos el archivo con los documentos
with open(filename, 'r', encoding='utf-8') as data:
    document_words = data.read().lower()

# aquí se crea el vocabulario de los documentos y la frecuencia de cada término
d = {}
for word in re.findall(r'[a-zñáíúéó]+', document_words):
    if word not in stop_words and len(word) > 2 and len(word) < 20:
        d[word] = d.get(word, 0) + 1
vocabulary = [word for word in d]

# Aqui se crea la matriz de term-document-frecuency
tdfm = []
with open(filename, 'r', encoding='utf-8') as data:
    for line in data:
        line = line.lower()
        tokens = re.findall(r'[a-zñáíúéó]+', line)
        vector = []
        for word in vocabulary:
            vector.append(tokens.count(word))
        tdfm.append(vector)
tdfm = np.array(tdfm)

# Aqui creo el archivo nuevo con las n palabras más frecuentes de cada documento
n = 8
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
with open(filename2, 'r', encoding='utf-8') as data:
    labels = data.readlines()
with open('labels_kw.txt', 'w', encoding='utf-8') as doc:
    temp = int(n/4)
    for label in labels:
        for i in range(temp):
            doc.write("%s" % label)

# Computar la similaridad
# calcular idf
idf = {}
for element in d:
    idf[element] = log((1+tdfm.shape[0]) / (1+d[element]) + 1)
# calcular tfidf
tfidf = []
for row in tdfm:
    # print(row)
    a = []
    for i, element in enumerate(row):
        a.append(row[i] * idf[vocabulary[i]])
    tfidf.append(a)
    
# normalizar tfidf
tfidf_norm = []
for row in tfidf:
    a = 0
    for element in row:
        a = a + element**2
    new_row = []
    for element in row:
        new_row.append(element / a**(1/2))
    tfidf_norm.append(new_row)

# calculamos el term frequency de las keywords
keywords_tf = []
with open('keywords.txt', 'r', encoding='utf-8') as text:
    for line in text:
        tokens = re.findall(r'[a-zñáíúéó]+', line)
        vector = []
        for word in vocabulary:
            vector.append(tokens.count(word))
        keywords_tf.append(vector)
keywords_tf = np.array(keywords_tf)

# calculamos la multiplicacion de idf x keywords_tf
keywords_tfidf = []
for row in keywords_tf:
    a = []
    for i, element in enumerate(row):
        a.append(row[i] * idf[vocabulary[i]])
    keywords_tfidf.append(a)

# normalizar keywords_tfidf
keywords_tfidf_norm = []
for row in keywords_tfidf:
    a = 0
    for element in row:
        a = a + element**2
    new_row = []
    for element in row:
        new_row.append(element / a**(1/2))
    keywords_tfidf_norm.append(new_row)

# cantidad de distancias a encontrar
m = 3
# encontrar las distancias euclidianas
dist_euc_k_t = []
for row in keywords_tfidf_norm:
    row_dist = []
    for row2 in tfidf_norm:
        d = 0
        for i, element in enumerate(row2):
            d = d + (row[i] - element)**2
        row_dist.append(d**(1/2))
    dist_euc_k_t.append(row_dist)

# encontrar las m distancias más cortas y sus documentos.
# en este vector guardamos las m distancias de cada keyword ordenadas de menor a mayor
dist_eucl_sort = []
# en este vector guardamos los indices de las m distancias de arriba.
eucl_dist_inde = []
# iteramos para cada lista de distancias de cada keyword
for distances in dist_euc_k_t:
    ind = range(tdfm.shape[0])
    ind = np.array(ind)
    eucl_dist_inde.append(ind[np.argsort(distances)[:m]])
    dist = np.array(distances)
    dist_eucl_sort.append(dist[np.argsort(distances)[:m]])
    
# mostrar los resultados
with open('keywords.txt', 'r', encoding='utf-8') as doc:
    keywords = doc.readlines()
with open('labels_kw.txt', 'r', encoding='utf-8') as doc:
    labels_kw = doc.readlines()
with open(filename, 'r', encoding='utf-8') as doc:
    documents = doc.readlines()
with open(filename2, 'r', encoding='utf-8') as doc:
    labels = doc.readlines()

for i, frase in enumerate(keywords):
    print(frase.strip(), ' - ', labels_kw[i].strip())
    for x in range(m):
        print('\t', labels[eucl_dist_inde[i][x]].strip(),dist_eucl_sort[i][x])