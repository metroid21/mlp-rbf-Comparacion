import csv
# Librerias de redes neuronales para python
import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from pandas import *

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

#Se agrega los datos al diccionario de entrenamiento
for numClase in range(6):
    cantInstancias  = len(datosEntrena[numClase])
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
percMLP = MLPClassifier(hidden_layer_sizes=(30,20), activation='relu', solver='adam', 
    alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
    max_iter=2000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
    nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

percMLP.n_layers=4
percMLP.n_outputs_=3
percMLP.classes_=clases
percMLP.fit(entrenamiento,clases)

#Creacion de la Red Neuronal con RBF
percRBF = svm.SVC(kernel="rbf")
percRBF.fit(entrenamiento,clases)

"""print "** Datos de Prueba **"
print pruebas

print "** Clases Esperadas **"
print clasesEsperadas"""

resultadosMLP = percMLP.predict(pruebas)
resultadosRBF = percRBF.predict(pruebas)
#Imprimimos los resultados
"""
print "** Clasificando con el MLP **"
print resultadosMLP
print "** Clasificando con el SVC con RBF **"
print resultadosRBF"""

#Imprimimos resultados de pruebas por clase
indiceInicio = 0

for numClase in range(6):
    print "Clase ", (numClase+1)
    cantEsperadas = clasesEsperadas.count(numClase+1)
    listaResultClaseMLP = resultadosMLP[indiceInicio:(indiceInicio+cantEsperadas)]
    cantClasiCorrectMLP = np.count_nonzero(listaResultClaseMLP == (numClase+1))
    listaResultClaseRBF = resultadosRBF[indiceInicio:(indiceInicio+cantEsperadas)]
    cantClasiCorrectRBF = np.count_nonzero(listaResultClaseRBF == (numClase+1))    
    print "\tExito\tError"    
    print "MLP:\t", cantClasiCorrectMLP, "\t",(cantEsperadas-cantClasiCorrectMLP)
    print "RBF \t", cantClasiCorrectRBF,"\t",(cantEsperadas-cantClasiCorrectRBF)
    print "Total ", len(datosPrueba[numClase])
    print
    indiceInicio += cantEsperadas
