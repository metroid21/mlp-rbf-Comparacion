import csv
# Librerias de redes neuronales para python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from pandas import *

datos = [[],[],[],[],[],[]]
datosEntrena = [[],[],[],[],[],[]]
datosPrueba = [[],[],[],[],[],[]]
dicParaEntrenar = {}
dicParaPrueba = {}
clases = []

with open('datos.csv','r') as archivo:
    reader = csv.reader(archivo)
    for row in reader:
        lista = []
        cantAtrib = len(row)
        numClase = int(row[cantAtrib-1])
        #Si no se necesita tener la clase como atributo agregar -1 a range(numAtrib)
        for i in range(cantAtrib-1):
            #Para evitar agregar las instancias con datos faltantes
            if(row[i] == '?'):
                continue
            lista.append(int(row[i]))
        #Si tiene 33 significa que le falto un atributo por eso no se agrega
        if len(lista) != 33:
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
for j in range(34):
    dicParaEntrenar[str(j)] = []
    dicParaPrueba[str(j)] = []

#Se agrega los datos al diccionario de entrenamiento
for numClase in range(6):
    cantInstancias  = len(datosEntrena[numClase])
    for numInstancia in range(cantInstancias):
        #Agrega cada clase de las instancias de entrenamiento
        clases.append(numClase)
        instancia = datosEntrena[numClase][numInstancia]
        cantAtrib = len(instancia)
        for numAtrib in range(cantAtrib):
            dicParaEntrenar[str(numAtrib)].append(instancia[numAtrib])

#Se agrega datos para el diccionario de prueba
#De las seis clases
for numClase in range(6):
    cantInstancias  = len(datosPrueba[numClase])
    #Probara con dos instancias de cada clase
    for numInstancia in range(2):
        instancia = datosPrueba[numClase][numInstancia]
        cantAtrib = len(instancia)
        for numAtrib in range(cantAtrib):
            dicParaPrueba[str(numAtrib)].append(instancia[numAtrib])

entrenamiento = DataFrame(dicParaEntrenar)
pruebas = DataFrame(dicParaPrueba)

perc = MLPClassifier(hidden_layer_sizes=(30,20), activation='relu', solver='adam', 
	alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
	max_iter=2000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
	nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

perc.n_layers=4
perc.n_outputs_=3
perc.classes_=clases
perc.fit(entrenamiento,clases)

print perc.predict(pruebas)
