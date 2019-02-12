#ENRIQUEZ BALLESTEROS, JAIME. 2018

import tensorflow as tf
from tensorflow import keras
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt



IMAGE_DIMS = (28, 28)


def cargar_numeros_desde_mnist():
	mnist = keras.datasets.mnist
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
	#para dividir entre los posibles colores por cada pixel, y que quede entre un rango [0,1]
	train_images = train_images / 255.0

	return (train_images, train_labels)


def crear_modelo_numeros(train_images, train_labels):
	EPOCHS = 5
	#model 1
	model = keras.Sequential([
	    keras.layers.Flatten(),
	    keras.layers.Dense(128, activation=tf.nn.relu),
	    keras.layers.Dense(10, activation=tf.nn.softmax)
	])


	model.compile(optimizer=tf.train.AdamOptimizer(),
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])

	model.fit(train_images, train_labels, epochs=EPOCHS)
	return model

def prediccion_numeros(model, image):
	data=[]
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)
	data = np.array(data, dtype="float") / 255.0
	predictions = model.predict(data)
	return np.argmax(predictions[0])+1
