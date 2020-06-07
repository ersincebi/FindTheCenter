'''
conda install -c anaconda graphviz pydot
or
pip install pydot pydotplus graphviz
'''

import os
import sys
import tensorflow as tf
from tensorflow import keras
from tf2.Kfocusingtf2 import FocusedLayer1D

path = "./models/"
files = os.listdir(path)

for file in files:
	try:
		restored_keras_model = keras.models.load_model(path+file, custom_objects={'FocusedLayer1D': FocusedLayer1D})
	except:
		restored_keras_model = keras.models.load_model(path+file)

	keras.utils.plot_model(restored_keras_model, to_file="./plots/"+file+".png")