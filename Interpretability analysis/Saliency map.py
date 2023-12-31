import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# 加载数据
img_path = "C:\\Users\\DELL\\Desktop\\code\\dataset\\covid-xray\\Data\\test\\COVID19\\COVID19(465).jpg"
img = load_img(img_path, target_size=(299, 299))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.

# 加载模型
loaded_model = load_model('C:\\Users\\DELL\\Desktop\\data processing\\covid\\VGG19\\COVID VGG19.h5')

# 选择输出类的索引（这里假设是二元分类，索引为0或1）
class_idx = 0

img_tensor = tf.convert_to_tensor(img_array)

# 计算输入图像相对于模型输出的梯度
with tf.GradientTape() as tape:
    tape.watch(img_tensor)
    preds = loaded_model(img_tensor)
    top_class = preds[:, class_idx]

# 获取梯度
grads = tape.gradient(top_class, img_tensor)

# 计算每个像素的导数的绝对值
dgrad_abs = tf.math.abs(grads)

# 将梯度的绝对值标准化到最大值为1
dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + tf.keras.backend.epsilon())

# 可视化显著性地图
plt.imshow(grad_eval, cmap='jet')
plt.axis('off')
plt.show()
# 使用模型预测图像
preds = loaded_model.predict(img_array)

# 查找具有最大概率的类索引
predicted_class_index = np.argmax(preds[0])
class_labels = ['COVID2_CT', 'Normal_CT', 'pneumonia_CT']
print("Predicted:", class_labels[predicted_class_index])
