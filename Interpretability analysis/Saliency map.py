import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Load the data
img_path = "C:\\Users\\DELL\\Desktop\\code\\dataset\\covid-xray\\Data\\test\\COVID19\\COVID19(465).jpg"
img = load_img(img_path, target_size=(299, 299))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Load the model
loaded_model = load_model('C:\\Users\\DELL\\Desktop\\data processing\\covid\\VGG19\\COVID VGG19.h5')

# Select the index of the output class (assuming binary classification, index is 0 or 1)
class_idx = 0

img_tensor = tf.convert_to_tensor(img_array)

# Compute the gradient of the input image with respect to the model's output
with tf.GradientTape() as tape:
    tape.watch(img_tensor)
    preds = loaded_model(img_tensor)
    top_class = preds[:, class_idx]

# Get the gradients
grads = tape.gradient(top_class, img_tensor)

# Calculate the absolute value of the derivative for each pixel
dgrad_abs = tf.math.abs(grads)

# Normalize the absolute value of gradients to a maximum value of 1
dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
arr_min, arr_max = np.min(dgrad_max_), np.max(dgrad_max_)
grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + tf.keras.backend.epsilon())

# Visualize the saliency map
plt.imshow(grad_eval, cmap='jet')
plt.axis('off')
plt.show()

# Use the model to predict the image
preds = loaded_model.predict(img_array)

# Find the index of the class with the highest probability
predicted_class_index = np.argmax(preds[0])
class_labels = ['COVID2_CT', 'Normal_CT', 'pneumonia_CT']
print("Predicted:", class_labels[predicted_class_index])
