"""Load Image, predict intentional framing"""
import os
import glob
import conf
import urllib
import operator
import numpy as np
import pandas as pd
from xml.dom import minidom


from keras.preprocessing import image
from keras.models import model_from_json
from keras.applications.inception_v3 import preprocess_input

classes = {'art': 0,
	'candid': 1,
	'landscape': 2,
	'macro': 3,
	'media_capture': 4,
	'portrait': 5,
	'product_presentation': 6,
	'setting': 7,
	'social_event_private': 8,
	'social_event_public': 9,
	'structure': 10}
classes = sorted(classes.items(), key=operator.itemgetter(1))

def prepare_model():
	"""prepare pre-trained model for intents classification."""
	def load_model():
		"""load model"""
		with open(conf.MODEL_PATH, 'r') as f:
			loaded_model_json = f.read()
			return model_from_json(loaded_model_json)

	def load_weights(model):
		"""Load mode weights."""
		model.load_weights(conf.WEIGHT_PATH)
		return model

	def compile_model(model):
		"""Compile current model."""
		model.compile(optimizer='rmsprop',
			loss='categorical_crossentropy',
			metrics=['accuracy'])
		return model

	return compile_model(load_weights(load_model()))

def make_prediction(url, img_id, model, classes, from_path=None):
	"""Download store in temp, make prediction, remove."""
	def download_img(url, img_id):
		"""Download image."""
		if from_path:
			return from_path
		img_path = ''.join([TEMP_PATH, img_id, '.jpg'])
		urllib.urlretrieve(url, img_path)
		return img_path

	def pre_process(img_path):
		"""Preprocess image"""
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		return x

	def predict(model, x, classes):
		"""Predict class."""
		preds = model.predict(x)
		prob = 0
		label = ''
		for pred, c in zip(preds[0], classes):
			if pred > prob:
				prob = pred
				label = c
		return prob, label

	img_path = download_img(url, img_id)
	img = pre_process(img_path)
	prob, label = predict(model, img, classes)
	return prob, label



