import tensorflow as tf
import shap
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import load_model

# 加载数据
data_path = "C:\\Users\\DELL\\Desktop\\code\\dataset\\breast\\BreaKHis 400X\\train"
train_data_path = data_path
datagen = ImageDataGenerator(rescale=1 / 255,
                            rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            fill_mode='constant',
                            validation_split=0.2,
                            horizontal_flip=True,
                            vertical_flip=True,
                            zoom_range=0.2
                            )

train_generator = datagen.flow_from_directory(
    train_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
# 加载要解释的图像
img_path = "C:\\Users\\DELL\\Desktop\\code\\dataset\\choose\\malignant breast\\SOB_M_DC-14-2523-400-019.png"
img_array = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224)))
img_array = np.expand_dims(img_array, axis=0) / 255.


# 加载模型
loaded_model = load_model('breast VGG19.h5')
# 创建一个Deep SHAP解释器。注意，SHAP希望有背景数据来估计shap值。
background = train_generator.next()[0]  # 使用一些训练数据作为背景。如果需要，你可以使用多个批次。
explainer = shap.GradientExplainer(loaded_model, background)

# 计算SHAP值
shap_values = explainer.shap_values(img_array)

# 可视化SHAP值
shap.image_plot(shap_values, img_array)

# 使用模型预测图像。
preds = loaded_model.predict(img_array)

# 查找具有最大概率的类索引。
predicted_class_index = np.argmax(preds[0])
class_labels = ['COVID2_CT', 'Normal_CT', 'pneumonia_CT']
print("Predicted:", class_labels[predicted_class_index])
