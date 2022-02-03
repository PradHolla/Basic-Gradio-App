import os
import gradio as gr
import tensorflow as tf
import requests
import tensorflow as tf
import tensorflow_hub as hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

import numpy as np
import PIL.Image
import time
import functools
# inception_net = tf.keras.applications.MobileNetV2() # load the model

# # Download human-readable labels for ImageNet.
# response = requests.get("https://git.io/JJkYN")
# labels = response.text.split("\n")

# def classify_image(inp):
#   inp = inp.reshape((-1, 224, 224, 3))
#   inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
#   prediction = inception_net.predict(inp).flatten()
#   return {labels[i]: float(prediction[i]) for i in range(1000)}
def tensor_to_image(tensor):
	tensor = tensor*255
	tensor = np.array(tensor, dtype=np.uint8)
	if np.ndim(tensor)>3:
		assert tensor.shape[0] == 1
		tensor = tensor[0]
	return PIL.Image.fromarray(tensor)

image = gr.inputs.Image(shape=(224, 224))
st_img = gr.inputs.Image(shape=(224, 224))
label = gr.outputs.Image()

def load_img(path_to_img):
	max_dim = 512
	img = tf.io.read_file(path_to_img)
	img = tf.image.decode_image(img, channels=3)
	img = tf.image.convert_image_dtype(img, tf.float32)

	shape = tf.cast(tf.shape(img)[:-1], tf.float32)
	long_dim = max(shape)
	scale = max_dim / long_dim

	new_shape = tf.cast(shape * scale, tf.int32)

	img = tf.image.resize(img, new_shape)
	img = img[tf.newaxis, :]
	return img

content_image = load_img(image)
style_image = load_img(st_img)

def neural_transfer(content_image, style_image):
	hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
	stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
	tensor_to_image(stylized_image)


gr.Interface(fn=tensor_to_image, inputs=(content_image, style_image), outputs=label, interpretation="default").launch()