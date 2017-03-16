import csv

# Librerias de redes neuronales para python
import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn import mixture
from pandas import *
from scipy.stats import mode

datos = [[],[],[],[],[],[]]
datosEntrena = [[],[],[],[],[],[]]
datosPrueba = [[],[],[],[],[],[]]
dicParaEntrenar = {}
dicParaPrueba = {}
clases = []
clasesEsperadas = []

with open('datos.csv','r') as archivo:
    reader = csv.reader(archivo)
    for row in reader:
        lista = []
        cantAtrib = len(row)
        numClase = int(row[cantAtrib-1])
        #Si no se necesita tener la clase como atributo agregar -1 a range(numAtrib)
        for i in range(cantAtrib-2):
            #Para evitar agregar las instancias con datos faltantes
            if(row[i] == '?'):
                continue
            lista.append(int(row[i]))
        #Si tiene 33 significa que le falto un atributo por eso no se agrega
        if len(lista) != 32:
                datos[numClase-1].append(lista)
        
#Se separan en las intancias para entrenamiento y las de pruebas
for i in range(6):
    cantInstancias = len(datos[i])
    cantEntrenar = int(cantInstancias*0.7)
    for j in range(cantEntrenar):
        datosEntrena[i].append(datos[i][j])
    for j in range(cantEntrenar,cantInstancias):
        datosPrueba[i].append(datos[i][j])


#Se agrega un diccionario para cada clase para luego usar en DataFrame
for j in range(len(datos[0][0])):
    dicParaEntrenar[str(j)] = []
    dicParaPrueba[str(j)] = []

cantidadPorClase = [];

#Se agrega los datos al diccionario de entrenamiento
for numClase in range(6):
    cantInstancias  = len(datosEntrena[numClase])
    cantidadPorClase.append(cantInstancias)
    for numInstancia in range(cantInstancias):
        #Agrega cada clase de las instancias de entrenamiento
        clases.append(numClase+1)
        instancia = datosEntrena[numClase][numInstancia]
        cantAtrib = len(instancia)
        for numAtrib in range(cantAtrib):
            dicParaEntrenar[str(numAtrib)].append(instancia[numAtrib])

#Se agrega datos para el diccionario de prueba
#De las seis clases

dataFramePorClase = []
            
for numClase in range(6):
    cantInstancias  = len(datosPrueba[numClase])
    #Probara con dos instancias de cada clase
    for numInstancia in range(cantInstancias):
        clasesEsperadas.append(numClase+1)
        instancia = datosPrueba[numClase][numInstancia]
        cantAtrib = len(instancia)
        for numAtrib in range(cantAtrib):
            dicParaPrueba[str(numAtrib)].append(instancia[numAtrib])

entrenamiento = DataFrame(dicParaEntrenar)
pruebas = DataFrame(dicParaPrueba)

#Creacion de la Red Neuronal MLP
percMLP = MLPClassifier(hidden_layer_sizes=(30,20,10))

percMLP.n_outputs_=6
percMLP.classes_=clases
percMLP.fit(entrenamiento,clases)

#Creacion de la Red Neuronal con SVM
percSVM = svm.SVC(kernel="rbf")
percSVM.fit(entrenamiento,clases)

#Creacion de la red neuronal con mezcla gausiana
gmm = mixture.GaussianMixture(n_components=6,max_iter=250)
gmm.fit(entrenamiento,clases)

#Creacion de una red con kmeans
percKM = KMeans(n_clusters=6)
percKM.fit(entrenamiento,clases)

resultadosMLP = percMLP.predict(pruebas)
resultadosSVM = percSVM.predict(pruebas)
resultadosKM  = percKM.predict(pruebas)
resultadosG   = gmm.predict(pruebas)
resultadosG2  = gmm.predict(entrenamiento)

#Calculamos la posicion de cada centro en relacion de cada clase
labels_centros = []
sum = 0
for i in range(len(cantidadPorClase)):
    cantInst = cantidadPorClase[i]
    sum += cantInst
    list = percKM.labels_[sum-cantInst: sum]
    labels_centros.append(mode(list)[0][0])

labels_centros2 = []
sum2 = 0
for i in range(len(cantidadPorClase)):
    cantInst = cantidadPorClase[i]
    sum2 += cantInst
    list = resultadosG2[sum2-cantInst: sum2]
    labels_centros2.append(mode(list)[0][0])

print labels_centros
print resultadosKM

#Imprimimos resultados de pruebas por clase
indiceInicio = 0

for numClase in range(6):
    print "Clase ", (numClase+1)
    cantEsperadas = clasesEsperadas.count(numClase+1)
    listaResultClaseMLP = resultadosMLP[indiceInicio:(indiceInicio+cantEsperadas)]
    cantClasiCorrectMLP = np.count_nonzero(listaResultClaseMLP == (numClase+1))
    listaResultClaseSVM = resultadosSVM[indiceInicio:(indiceInicio+cantEsperadas)]
    cantClasiCorrectSVM = np.count_nonzero(listaResultClaseSVM == (numClase+1))    
    listaResultClaseKM  = resultadosKM[indiceInicio:(indiceInicio+cantEsperadas)]
    cantClasiCorrectKM  = np.count_nonzero(listaResultClaseKM == labels_centros[numClase])    
    listaResultClaseG = resultadosG[indiceInicio:(indiceInicio+cantEsperadas)]
    cantClasiCorrectG  = np.count_nonzero(listaResultClaseG == labels_centros2[numClase])    

    print "\tExito\tError"    
    print "MLP:\t", cantClasiCorrectMLP, "\t",(cantEsperadas-cantClasiCorrectMLP)
    #print "SVM \t", cantClasiCorrectSVM,"\t",(cantEsperadas-cantClasiCorrectSVM)
    print "KM \t",  cantClasiCorrectKM,"\t",(cantEsperadas-cantClasiCorrectKM)
    print "G \t",  cantClasiCorrectG,"\t",(cantEsperadas-cantClasiCorrectG)
    print "Total ", len(datosPrueba[numClase])
    print
    indiceInicio += cantEsperadas