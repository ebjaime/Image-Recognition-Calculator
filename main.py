#ENRIQUEZ BALLESTEROS, JAIME. 2018
import cv2
from reconocimiento_operaciones import crear_modelo_operaciones
from reconocimiento_operaciones import prediccion_operacion
from reconocimiento_numeros import cargar_numeros_desde_mnist
from reconocimiento_numeros import crear_modelo_numeros
from reconocimiento_numeros import prediccion_numeros
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
import sys
import operator

def split_image_into(input_path, n):
	images = []
	img = cv2.imread(input_path)
	imgheight, imgwidth, channels = img.shape
	width = imgwidth / n;
	for j in range(0,imgwidth,width):
		crop_img = img[imgheight/2 - width/2 :imgheight/2 + width/2, j:j+width]#
		images.append(crop_img)
	return images


def help_menu():
	print("\nOPCIONES")
	print("\n")
	print("(*) Entrenar modelo de operaciones con imagenes en directorio 'dir':")
	print("\tpython main.py -eo dir ...")
	print("\n(*) Entrenar modelo de numeros con imagenes de la base de datos mnist:")
	print("\tpython main.py -en ...")
	print("\n(*) Predecir operacion en fichero 'fich.jpg' con 'n' numero de elementos (tal que '3+3' -> 3 elementos):")
	print("\tpython main.py ... -op fich.jpg n")
	print("\n(*) Predecir operando en fichero 'fich.jpg':")
	print("\tpython main.py ... -operand fich.jpg")
	print("\n(*) Predecir numero en fichero 'fich.jpg':")
	print("\tpython main.py ... -num fich.jpg")
	print("\n(*) Predecir operacion en fichero 'fich.jpg' con 'n' numero de elementos y resolver:")
	print("\tpython main.py ... -op fich.jpg n -r")
	return

def get_operator_fn(op):
    return {
        '+' : operator.add,
        '-' : operator.sub,
        'x' : operator.mul,
        '/' : operator.div,
        }[op]


def solve_operation(pred):
	num=[]
	ops=[]
	j=0
	while j<len(pred):
		if(j%2 ==0):
			num.append(int(pred[j]))
		else:
			ops.append(get_operator_fn(pred[j]))
		j+=1

	# Multiplicacion y division tienen prioridad
	j=0
	elim=0
	while j<(len(ops)-elim):
		if(ops[j]==operator.mul or ops[j]==operator.div):
			num[j]=ops[j](num[j], num[j+1])
			ops.remove(ops[j])
			num.remove(num[j+1])
		else:
			j+=1
	# Suma y resta
	j=0
	while len(num)!=1:
		num[j]=ops[j](num[j], num[j+1])
		ops.remove(ops[j])
		num.remove(num[j+1])


	return num[0];

def main():

	if(len(sys.argv)<2):
		help_menu()
		return 1
	if(sys.argv[1] == "-h"):
		help_menu()
		return 1

	i=1
	while i<len(sys.argv):
		if(sys.argv[i] == "-eo"):
			print("\nEntrenando modelo_operaciones\n")
			modelo_operaciones=crear_modelo_operaciones(sys.argv[i+1]) #buscar_y_descargar_imagenes
		if(sys.argv[i] == "-en"):
			print("\nCargando base de datos con imagenes de numeros de mnist\n")
			numeros_data, numeros_labels = cargar_numeros_desde_mnist()
			print("\nCargada base de datos mnist")
			print("\nEntrenando modelo_numeros\n")
			modelo_numeros = crear_modelo_numeros(numeros_data, numeros_labels)
		i+=1

	option = "NONE"

	i=1
	while i<len(sys.argv):
		if(sys.argv[i] == "-op"):
			#sys.argv[i+1] = name of image
			#sys.argv[i+2] = number of elements to be recognized
			pred = []
			images=split_image_into(sys.argv[i+1], int(sys.argv[i+2]))
			j=0
			while j<int(sys.argv[i+2]):
				if(j%2 == 0):
					pred.append(prediccion_numeros(modelo_numeros,images[j]))
				else:
					pred.append(prediccion_operacion(modelo_operaciones, images[j]))
				j+=1
			option = "op"
			break

		elif(sys.argv[i] == "-operand"):
			images = sys.argv[i+1]
			pred = prediccion_operacion(modelo_operaciones, cv2.imread(images))
			option = "operand"
			break

		elif(sys.argv[i] == "-num"):
			images = sys.argv[i+1]
			pred = prediccion_numeros(modelo_numeros, cv2.imread(images))
			option = "num"
			break

		i+=1

	if(option == "NONE"):
		print("No se ha seleccionado ninguna opcion. No se va a predecir nada")

	if(option == "op"):
		i=0
		while i<len(pred):
			print(pred[i])
			i+=1
	if(option == "operand"):
		print("Prediccion de operacion: " + pred)
	if(option == "num"):
		print("Prediccion de numero: " + str(pred))


	# Opcion de resolver operacion
	i=1
	result = None
	while i<len(sys.argv):
		if(option == "op") and (sys.argv[i]=="-r"):
			result=solve_operation(pred)
			break
		i+=1
	if (result != None):
		print("=")
		print result
	return 1

main()
