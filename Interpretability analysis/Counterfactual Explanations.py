import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_eager_execution()
from tensorflow.keras.models import load_model
from alibi.explainers import CounterFactualProto

# Load the model
loaded_model = load_model('New COVID19 ResNet101.h5')

# Load and preprocess the image
img_path = "C:\\Users\\DELL\\Desktop\\code\\dataset\\covid-xray\\Data\\test\\COVID19\\COVID19(465).jpg"  # Replace with your image path
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Initialize the counterfactual explainer
shape = img_array.shape
cf = CounterFactualProto(loaded_model, shape, theta=10., max_iterations=100,
                         feature_range=(img_array.min(), img_array.max()))

# Generate the counterfactual
explanation = cf.explain(img_array)

# Counterfactual result
counterfactual = explanation.cf['X']
if counterfactual is not None:
    # Calculate the difference between the original image and the counterfactual
    difference = np.abs(img_array - counterfactual)
    contrast_factor = 10  # Experiment with different factors to get the best result
    enhanced_difference = np.clip(
        contrast_factor * (difference - difference.min()) / (difference.max() - difference.min()), 0, 1)

    # Create a figure to show the original image, counterfactual image, and difference
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img_array[0])
    plt.axis('off')
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(counterfactual.reshape(224, 224, 3))
    plt.axis('off')
    plt.title('Counterfactual Image')

    plt.subplot(1, 3, 3)
    plt.imshow(enhanced_difference.reshape(224, 224, 3))
    plt.axis('off')
    plt.title('Difference')

    plt.show()
else:
    print("No counterfactual found!")
