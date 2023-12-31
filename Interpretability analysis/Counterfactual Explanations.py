import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_eager_execution()
from tensorflow.keras.models import load_model
from alibi.explainers import CounterFactualProto

# 加载模型
loaded_model = load_model('New COVID19 ResNet101.h5')

# 加载图像并进行预处理
img_path = "C:\\Users\\DELL\\Desktop\\code\\dataset\\covid-xray\\Data\\test\\COVID19\\COVID19(465).jpg"# 替换为您的图像路径
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 初始化反事实解释器
shape = img_array.shape
cf = CounterFactualProto(loaded_model, shape, theta=10., max_iterations=100,
                         feature_range=(img_array.min(),img_array.max()))

# 计算反事实
explanation = cf.explain(img_array)

# 反事实的结果
counterfactual = explanation.cf['X']
if counterfactual is not None:
    # 计算原始图像与反事实之间的差异
    difference = np.abs(img_array - counterfactual)
    contrast_factor = 10  # 试验不同的因子以获得最佳结果
    enhanced_difference = np.clip(
        contrast_factor * (difference - difference.min()) / (difference.max() - difference.min()), 0, 1)

    # 创建一个图形来显示原始图像、反事实图像以及差异
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