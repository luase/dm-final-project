from nltk.corpus import stopwords
from math import log
import numpy as np
import re

"""
    Variables globales de configuración del programa
"""
# se definen las stopwords para español
stop_words = stopwords.words('spanish')
stop_words = set(stop_words)
# documents_filename es el nombre del archivo con los documentos
documents_filename = 'data.txt'
# labels_filename es el nombre del archivo con las etiquetas de los documentos
labels_filename = 'labels.txt'
# 'keywords_filename' es el nombre del archivo de salida para guardar las keywords.
keywords_filename = 'keywords.txt'
# 'keywords_label' es el nombre del archivo de salida para guardar las labels.
keywords_labels = 'labels_kw.txt'
# n es la cantidad de palabras más frecuentes que nos interesa obtener
# de cada documento, debe ser un múltiplo de 4 preferentemente.
n = 16
# m es la cantidad de documentos similares que deseamos obtener por cada keyword.
m = 10

"""
    Aqui están definidas las funciones que se usan en el programa
"""
# funcion para crear un diccionario
def dict_gen(filename):
    # abrimos el archivo con los documentos
    with open(filename, 'r', encoding='utf-8') as data:
        document_words = data.read().lower()
    # aquí se crea el diccionario de los documentos y  se calcula la frecuencia de cada término
    d = {}
    for word in re.findall(r'[a-zñáíúéó]+', document_words):
        if word not in stop_words and len(word) > 2 and len(word) < 20:
            d[word] = d.get(word, 0) + 1
    return d

# funcion para generar Term-Document-Frequency Matrix
def tdfm_gen(filename, vocabulary):
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
    return tdfm

# funcion para crear el archivo keywords_filename
def keywordstxt_gen(filename, tdfm, vocabulary):
    # 'indexes' almacena los indices del vocabulario
    indexes = np.array(range(len(vocabulary)))
    # 'documents_keywords' guarda todas las keywords de todos los documentos ordenadas de menor a mayor frecuencia
    documents_keywords = []
    for row in tdfm:
        # para cada renglón de la matriz de frecuencias
        keywords = []
        for index in indexes[np.argsort(row)[-n:]]:
            # encontrar las n palabras más repetidas
            keywords.append(vocabulary[index])
        documents_keywords.append(keywords)
    # 'keywords_file' almacena las keywords que irán en un archivo ordenadas de acuerdo a la definicion del problema
    keywords_file = []
    for document_keywords in documents_keywords:
        # generar los n/4 documentos nuevos con las palabras más frecuentes de cada documento
        temp = int(n/4)
        for i in range(temp):
            keywords = []
            keywords.append(document_keywords[-((i*2)+1)])
            keywords.append(document_keywords[-((i*2)+2)])
            keywords.append(document_keywords[(i*2)+1])
            keywords.append(document_keywords[(i*2)])
            keywords_file.append(keywords)
    # guardar los nuevos documentos
    with open(filename, 'w', encoding='utf-8') as doc:
        for keys in keywords_file:
            for key in keys:
                doc.write("%s " % key)
            doc.write("\n")
    # generar el archivo con las labels nuevas
    # abrimos el archivo con las etiquetas
    with open(labels_filename, 'r', encoding='utf-8') as data:
        labels = data.readlines()
    with open('labels_kw.txt', 'w', encoding='utf-8') as doc:
        temp = int(n/4)
        for label in labels:
            for i in range(temp):
                doc.write("%s" % label)

# funcion para generar idf
def idf_gen(dictionary, mat):
    idf = {}
    for element in dictionary:
        # calcular idf
        idf[element] = log((1+mat.shape[0]) / (1+dictionary[element]) + 1)
    return idf

# funcion para multiplicar matriz 'mat' por 'idf'
def mat_idf(mat, idf, vocabulary):
    tdfm_idf = []
    for row in mat:
        a = []
        for i, element in enumerate(row):
            # calcular tfidf
            a.append(element * idf[vocabulary[i]])
        tdfm_idf.append(a)
    return tdfm_idf

# función para normalizar matriz 'mat'
def matNorm(mat):
    mat_norm = []
    for row in mat:
        a = 0
        # encontramos la sumatoria de los cuadrados
        for element in row:
            a = a + element**2
        new_row = []
        # dividimos cada elemento entre la raiz cuadrada de la sumatoria previa
        for element in row:
            new_row.append(element / a**(1/2))
        # el nuevo vector se añade a la matriz normalizada
        mat_norm.append(new_row)
    return mat_norm

# funcion para encontrar distancia euclidiana inversa entre dos matrices.
def matInvEuclDis(mat_a, mat_b):
    invEuclDis = []
    # se itera por cada renglón de la 'mat_a'
    for row in mat_a:
        row_invEuclDist = []
        # se itera por cada renglón de la 'mat_b'
        for row2 in mat_b:
            d = 0
            # se itera por cada elemento de cada renglón de 'mat_b'
            for i, element in enumerate(row2):
                # se calcula la sumatoria de los cuadrados
                d = d + (row[i] - element)**2
            # se guarda el valor de la inversa de la raiz de la sumatoria.
            row_invEuclDist.append(1.0/(d**(1/2)))
        # se guarda la lista de distancias del renglón de 'mat_a' respecto a todos los renglones de 'mat_b'
        invEuclDis.append(row_invEuclDist)
    return invEuclDis

# funcion para encontrar la similitud de cosenos entre dos matrices.
def matCosSim(mat_a, mat_b):
    cosSim = []
    # se itera por cada renglón de la 'mat_a'
    for row in mat_a:
        row_cosSim = []
        # se itera por cada renglón de la 'mat_b'
        for row2 in mat_b:
            s = 0
            # se itera por cada elemento de cada renglón de 'mat_b'
            for i, element in enumerate(row2):
                # se calcula el producto punto de los dos renglones
                s = s + (row[i] * element)
            # se guarda el valor del producto punto.
            row_cosSim.append(s)
        # se guarda la lista de similitudes por coseno del renglón de 'mat_a' respecto a todos los renglones de 'mat_b'
        cosSim.append(row_cosSim)
    return cosSim

# función para guardar los m valores más grandes.
def getBest(m, results):
    # 'results_sortd' guarda los m valores más grandes en orden ascendente
    results_sortd = []
    # 'results_index' guarda el índice de cada documento de 'results_sortd'
    results_index = []
    for row in results:
        temp_index = np.array(range(len(results[0])))
        results_index.append(temp_index[np.argsort(row)[-m:]])
        row_copy = np.array(row)
        results_sortd.append(row_copy[np.argsort(row)[-m:]])
    return results_sortd, results_index

# función para imprimir los resultados.
def printRes(kw_fn, kw_lb, d_lb, ind, val):
    with open(kw_fn, 'r', encoding='utf-8') as doc:
        keywords = doc.readlines()
    with open(kw_lb, 'r', encoding='utf-8') as doc:
        labels_kw = doc.readlines()
    with open(d_lb, 'r', encoding='utf-8') as doc:
        labels = doc.readlines()

    for i, frase in enumerate(keywords):
        print(frase.strip(), ' - ', labels_kw[i].strip())
        for x in range(m):
            print('\t', labels[ind[i][-(x+1)]].strip(), val[i][-(x+1)])

"""
    main, el orden lógico del programa
"""
if __name__ == "__main__":
    """
        variables iniciales para el análisis de similaridad
    """
    # crear un diccionario
    d = dict_gen(documents_filename)
    # crear el vocabulario
    vocabulary = [word for word in d]
    """
        Aqui se crea la matriz de term-document-frecuency de los documentos y de las keywords
    """
    tdfm = tdfm_gen(documents_filename, vocabulary)
    # crear el archivo keywords_filename
    keywordstxt_gen(keywords_filename, tdfm, vocabulary)
    keywords_tdfm = tdfm_gen(keywords_filename, vocabulary)
    """
        variables necesarias para computar la similaridad
    """
    # idf utilizando logarítmos
    idf = idf_gen(d, tdfm)
    # productio cruz tdfm x idf para los documentos y keywords
    tdfm_idf = mat_idf(tdfm, idf, vocabulary)
    keywords_tdfm_idf = mat_idf(keywords_tdfm, idf, vocabulary)
    # normalización de matrices
    tdfm_idf_norm = matNorm(tdfm_idf)
    keywords_tdfm_idf_norm = matNorm(keywords_tdfm_idf)
    """
        cómputo de distancias euclidianas inversas y zimilitud de cosenos
    """
    invEuclDis = matInvEuclDis(keywords_tdfm_idf, tdfm_idf_norm)
    cosSimilarity = matCosSim(keywords_tdfm_idf_norm, tdfm_idf_norm)
    """
        Se satisface la definición del problema:
            "Obtener los m documentos más similares por similitud de
             coseno y distancia euclidiana inversa."
    """
    iedSortd, iedIndex = getBest(m, invEuclDis)
    csSortd, csIndex = getBest(m, cosSimilarity)
    """
        Guardar los resultados a un archivo txt
    """
    printRes(keywords_filename, keywords_labels, labels_filename, iedIndex,iedSortd)