import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the model
model = keras.models.load_model('1101.h5')

# Set the name of the last convolutional layer
last_conv_layer_name = "relu"  # Replace with the name of the last convolutional layer in your model
# Label list
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


# Grad-CAM function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that outputs the last convolutional layer and the original model's output given the model input
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the class prediction
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the feature map output
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # The average of the derivatives for each feature map, these are the Grad-CAM weights
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # We apply the weights to the output of the last convolutional layer
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # To make the heatmap visible, we normalize it
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()


def preprocess_image(img_path):
    # Use the same preprocessing steps as when training the model
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Ensure it matches the preprocessing during training
    return img


def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Resize the heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Display the superimposed image
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()


# Main function
def main(img_path):
    img_array = preprocess_image(img_path)

    # Predict the image
    preds = model.predict(img_array)
    pred_index = tf.argmax(preds[0])
    print(f"Predicted class: {labels[pred_index]} with confidence {preds[0][pred_index]:.2f}")

    # Generate the Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)

    # Display the Grad-CAM heatmap
    save_and_display_gradcam(img_path, heatmap)


img_path = "C:\\Users\\DELL\\Desktop\\code\\dataset\\image(3).jpg"
main(img_path)
