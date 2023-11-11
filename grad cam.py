import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt

m = keras.models.load_model('densnet.h5')
last_conv_layer_name = "relu"
#last_conv_layer_name = "block5_conv3"
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
img_path = "C:\\Users\\DELL\\Desktop\\code\\dataset\\image(13).jpg"


def gradCam(image, true_label, layer_conv_name):
    model_grad = tf.keras.models.Model(inputs=m.input,
                                       outputs=[m.get_layer(layer_conv_name).output, m.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = model_grad(image)
        tape.watch(conv_output)
        loss = tf.keras.losses.binary_crossentropy(true_label, predictions)

    grad = tape.gradient(loss, conv_output)
    grad = K.mean(tf.abs(grad), axis=(0, 1, 2))
    conv_output = np.squeeze(conv_output.numpy())

    for i in range(conv_output.shape[-1]):
        conv_output[:, :, i] = conv_output[:, :, i] * grad[i]

    heatmap = tf.reduce_mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    return np.squeeze(heatmap)


def display_heatmap(image_path, heatmap):
    img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM Heatmap")
    plt.show()


def main(img_path):
    # Preprocess the image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    predictions = m.predict(img_array)
    predicted_class = labels[np.argmax(predictions)]

    # Generate heatmap
    heatmap = gradCam(img_array, predictions, last_conv_layer_name)

    # Display results
    print(f"Predicted Class: {predicted_class}")
    display_heatmap(img_path, heatmap)


main(img_path)
