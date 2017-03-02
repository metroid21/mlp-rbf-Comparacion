# Librerias de redes neuronales para python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from pandas import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

clases = [0,1,2]

entrenamiento = DataFrame({
'0' : [0,0,0],
'1' : [1,0,1],
'2' : [1,1,1],
'3' : [1,0,1],
'4' : [0,0,0],
'5' : [1,0,1],
'6' : [0,1,0],
'7' : [0,1,0],
'8' : [0,0,0],
'9' : [1,0,1],
'10' : [1,0,0],
'11' : [0,0,0],
'12' : [0,1,0],
'13' : [1,0,0],
'14' : [1,0,1],
'15' : [1,0,0],
'16' : [0,0,0],
'17' : [1,1,0],
'18' : [0,0,1],
'19' : [1,0,0],
'10' : [1,0,0],
'21' : [1,0,0],
'22' : [0,1,1],
'23' : [0,0,0],
'24' : [1,0,0],
'25' : [1,0,0],
'26' : [0,0,1],
'27' : [0,1,0],
'28' : [0,0,0],
'29' : [1,0,0],
'30' : [0,0,1],
'31' : [1,0,1],
'32' : [1,1,1],
'33' : [1,1,1],
'34' : [0,1,1]
	})

prueba = DataFrame({
'0' : [0],
'1' : [0],
'2' : [1],
'3' : [0],
'4' : [0],
'5' : [0],
'6' : [1],
'7' : [1],
'8' : [0],
'9' : [0],
'10' : [0],
'11' : [0],
'12' : [1],
'13' : [0],
'14' : [0],
'15' : [0],
'16' : [0],
'17' : [1],
'18' : [0],
'19' : [0],
'10' : [0],
'21' : [0],
'22' : [1],
'23' : [0],
'24' : [0],
'25' : [0],
'26' : [0],
'27' : [1],
'28' : [0],
'29' : [0],
'30' : [0],
'31' : [0],
'32' : [1],
'33' : [1],
'34' : [1]
	})

perc = MLPClassifier(hidden_layer_sizes=(30,20), activation='relu', solver='adam', 
	alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
	max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
	nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

perc.n_layers=4
perc.n_outputs_=3
perc.classes_=clases

perc.fit(entrenamiento,clases)
print perc.predict(prueba)
