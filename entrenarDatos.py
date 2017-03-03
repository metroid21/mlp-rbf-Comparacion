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
        instancia = datosEntrena[numClase][numInstancia]
        cantAtrib = len(instancia)
        for numAtrib in range(cantAtrib):
            dicParaEntrenar[str(numAtrib)].append(instancia[numAtrib])

#Se agrega datos para el diccionario de prueba
for numClase in range(6):
    cantInstancias  = len(datosPrueba[numClase])
    for numInstancia in range(cantInstancias):
        instancia = datosPrueba[numClase][numInstancia]
        cantAtrib = len(instancia)
        for numAtrib in range(cantAtrib):
            dicParaPrueba[str(numAtrib)].append(instancia[numAtrib])

entrenamiento = DataFrame(dicParaEntrenar)
pruebas = DataFrame(dicParaPrueba)
        
    


