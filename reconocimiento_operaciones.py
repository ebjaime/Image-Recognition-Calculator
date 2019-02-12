#ENRIQUEZ BALLESTEROS, JAIME. 2018

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from random import seed

IMAGE_DIMS = (96, 96)
#function which is used as pipeline between raw data and data which is going to be used by tensorflow
def transformacion_operaciones_a_datalabels(directorio_dataset):


	imagePaths = []

	for i in range(len(list(paths.list_images(directorio_dataset+"/dataset/scanner/plus_sign")))):
		(imagePaths).append(sorted(list(paths.list_images(directorio_dataset+"/dataset/scanner/plus_sign")))[i])
	for i in range(len(list(paths.list_images(directorio_dataset+"/dataset/scanner/minus_sign")))):
		(imagePaths).append(sorted(list(paths.list_images(directorio_dataset+"/dataset/scanner/minus_sign")))[i])
	for i in range(len(list(paths.list_images(directorio_dataset+"/dataset/scanner/mult_sign")))):
		(imagePaths).append(sorted(list(paths.list_images(directorio_dataset+"/dataset/scanner/mult_sign")))[i])
	for i in range(len(list(paths.list_images(directorio_dataset+"/dataset/scanner/div_sign")))):
		(imagePaths).append(sorted(list(paths.list_images(directorio_dataset+"/dataset/scanner/div_sign")))[i])


	print("Loaded all image's paths")

	random.seed(42)
	random.shuffle(imagePaths)

	print("Shuffled all image's paths\n")

	data = []
	labels = []

	for imagePath in imagePaths:
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
		image = img_to_array(image)
		data.append(image)

		label = imagePath.split(os.path.sep)[-2]
		labels.append(label)

	data = np.array(data, dtype="float") / 255.0
	labels = np.array(labels)
	return data, labels

#function which is used to create a model from a dataset indside the directorio_dataset directory
def crear_modelo_operaciones(directorio_dataset):
	EPOCHS = 50
	data, labels = transformacion_operaciones_a_datalabels(directorio_dataset)

	# X -> data; Y -> labels
	#if we want to divide the dataset between a train dataset and a test dataset
	#(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

	#model 2
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(96, 96, 1)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4))
	model.add(Activation('sigmoid'))

	model.compile(optimizer='adam',
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])

	#We transform the labels to ints for the model to be able to process it
	dict_labels = {'plus_sign':0, 'minus_sign':1, 'mult_sign':2, 'div_sign':3}



	labels_ints = []
	for i in range(len(labels)):
		labels_ints.append(dict_labels[labels[i]])


	#The model fits the data into the model
	model.fit(data, np.asarray(labels_ints), epochs=EPOCHS)
	return model

def prediccion_operacion(model, image):
	data=[]
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)
	data = np.array(data, dtype="float") / 255.0

	dict_labels_inv = {0:'+', 1:'-', 2:'x', 3:'/'}
	predictions = model.predict(data)
	return dict_labels_inv[np.argmax(predictions[0])]
