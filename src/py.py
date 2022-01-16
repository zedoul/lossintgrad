import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),
                                          include_top=True,
                                          weights='imagenet')

def load_imagenet_labels(file_path):
  labels_file = tf.keras.utils.get_file('ImageNetLabels.txt', file_path)
  with open(labels_file) as reader:
    f = reader.read()
    labels = f.splitlines()
  return np.array(labels)
imagenet_labels = load_imagenet_labels('https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

def read_image(file_name):
  image = tf.io.read_file(file_name)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)  
  image = tf.image.resize_with_pad(image, target_height=224, target_width=224)
  return image

img = {'Peacock':'Peacock.jpg'}
img_name_tensors = {name: read_image(img_path) for (name, img_path) in img.items()}


plt.figure(figsize=(5, 5))
ax = plt.subplot(1, 1, 1)
ax.imshow(img_name_tensors['Peacock'])
ax.set_title("Image")
ax.axis('off')
plt.tight_layout()


def top_k_predictions(img, k=3):
  image = tf.expand_dims(img, 0)
  predictions = model(image)
  probs = tf.nn.softmax(predictions, axis=-1)
  top_probs, top_idxs = tf.math.top_k(input=probs, k=k)
  top_labels = np.array(tuple(top_idxs[0]) )
  return top_labels, top_probs[0]#Display the image with top 3 prediction from the model
plt.imshow(img_name_tensors['Peacock'])
plt.title(name, fontweight='bold')
plt.axis('off')
plt.show()


pred_label, pred_prob = top_k_predictions(img_name_tensors['Peacock'])
for label, prob in zip(pred_label, pred_prob):
    print(f'{imagenet_labels[label+1]}: {prob:0.1%}')

