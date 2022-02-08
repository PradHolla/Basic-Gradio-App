import gradio as gr
import tensorflow as tf
import requests
import numpy as np
inception_net = tf.keras.applications.MobileNetV2() # load the model

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")
class_names = ["Early Blight", "Late Blight", "Healthy"]

# def classify_image(inp):
#   inp = inp.reshape((-1, 224, 224, 3))
#   inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
#   prediction = inception_net.predict(inp).flatten()
#   return {labels[i]: float(prediction[i]) for i in range(1000)}
model = tf.keras.models.load_model("potatoes.h5")
image = gr.inputs.Image(shape=(256, 256))
label = gr.outputs.Label(num_top_classes=3)

def classify_imagee(inp):
	inp = inp.reshape((-1, 256, 256, 3))
	inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
	prediction = model.predict(inp)
	prediction = np.argmax(prediction[0])
	return class_names[prediction]

gr.Interface(fn=classify_imagee, inputs=image, outputs="text", interpretation="default", live=True).launch()