import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import EfficientNetB7
import sys
import gym
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import os
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Confirm TensorFlow can see the GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Set memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

def load_data(datasetfolder, image_data_generator):
    dataflowtraining = image_data_generator.flow_from_directory(
        directory=datasetfolder,
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=16,
        shuffle=True,
        subset='training')

    dataflowvalidation = image_data_generator.flow_from_directory(
        directory=datasetfolder,
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=16,
        shuffle=True,
        subset='validation')


    return dataflowtraining, dataflowvalidation


def plot_sample_images(dataflowvalidation):
    images, labels = dataflowvalidation.next()
    plt.figure(figsize=(12, 12))
    for i in range(16):
        plt.subplot(4, 4, (i + 1))
        plt.imshow(images[i])
        plt.title(np.argmax(labels[i]))
    plt.show()

def build_model():
    # Load EfficientNetB0 with pre-trained ImageNet weights
    basemodel = EfficientNetB7(weights='imagenet', include_top=False,
                               input_shape=(224, 224, 3), pooling=None)

    # Add new top layers
    x = tf.keras.layers.GlobalAveragePooling2D()(basemodel.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    # Create the model
    m = tf.keras.models.Model(inputs=basemodel.input, outputs=x)

    # Compile the model
    m.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')])
    return m


def plot_history(hist):
    plt.figure(figsize=(12, 6))
    metrics = ['loss', 'precision', 'recall', 'accuracy']
    for i in range(4):
        plt.subplot(2, 2, (i + 1))
        plt.plot(hist.history[metrics[i]], label=metrics[i])
        plt.plot(hist.history['val_{}'.format(metrics[i])], label='val_{}'.format(metrics[i]))
        plt.legend()
    plt.show()


class DataAugmentationEnv(gym.Env):
    def __init__(self, datasetfolder):
        self.best_val_loss = 99
        super(DataAugmentationEnv, self).__init__()

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(4,))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.datasetfolder = datasetfolder

        ge = ImageDataGenerator(rescale=1 / 255,
                            rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            fill_mode='constant',
                            validation_split=0.3,
                            horizontal_flip=True,
                            vertical_flip=True,
                            zoom_range=0.2,)

        self.dataflowtraining, self.dataflowvalidation = load_data(self.datasetfolder, ge)

    def reset(self):
        return np.array([0.5])

    def step(self, action):
        updated_generator = ImageDataGenerator(
            rotation_range=action[0] * 0.3,
            width_shift_range=action[1] * 0.1,
            height_shift_range=action[2] * 0.1,
            zoom_range=action[3] * 0.3,
            rescale=1 / 255,
            fill_mode='constant',
            validation_split=0.2,
            horizontal_flip=True,
            vertical_flip=True,
        )

        # 使用更新后的生成器加载数据
        self.dataflowtraining, self.dataflowvalidation = load_data(self.datasetfolder, updated_generator)

        m = build_model()

        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(patience=8, monitor='val_loss', mode='min', restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=6, monitor='val_loss', mode='min', factor=0.1)
        ]

        hist = m.fit(
            self.dataflowtraining,
            epochs=100,
            validation_data=self.dataflowvalidation,
            verbose=0,
            callbacks=callbacks_list
        )

        # 其余代码保持不变

        print("loss: {:.4f} - accuracy: {:.4f} - precision: {:.4f} - recall: {:.4f} - val_loss: {:.4f} - val_accuracy: {:.4f} - val_precision: {:.4f} - val_recall: {:.4f} - lr: {:.1e}".format(
            hist.history['loss'][-1], hist.history['accuracy'][-1], hist.history['precision'][-1], hist.history['recall'][-1],
            hist.history['val_loss'][-1], hist.history['val_accuracy'][-1], hist.history['val_precision'][-1], hist.history['val_recall'][-1],
            float(tf.keras.backend.get_value(m.optimizer.learning_rate))
        ))

        val_loss = np.min(hist.history['val_loss'])
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            m.save('efiicientnetb7.h5')
            print('save the model')
        reward = -val_loss  # 注意这里取了负值，因为损失越低越好。
        done = True
        return np.array([0.5]), reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

def build_actor(env):
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(16, activation="relu"))
    actor.add(Dense(16, activation="relu"))
    actor.add(Dense(16, activation="relu"))
    actor.add(Dense(env.action_space.shape[0], activation="linear"))
    return actor

def build_critic(env):
    action_input = tf.keras.layers.Input(shape=(env.action_space.shape[0],))
    observation_input = tf.keras.layers.Input(shape=(1,) + env.observation_space.shape)
    flattened_observation = Flatten()(observation_input)
    x = tf.keras.layers.Concatenate()([action_input, flattened_observation])
    x = Dense(32, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(1, activation="linear")(x)
    critic = tf.keras.models.Model(inputs=[action_input, observation_input], outputs=x)
    return action_input, critic


def main():
    # 保存当前的标准输出流
    original_stdout = sys.stdout

    # 打开文件，如果文件不存在则创建，如果存在则覆盖
    with open('efficientnetb7 result.txt', 'w') as f:
        # 将标准输出重定向到文件
        sys.stdout = f
        datasetfolder = 'C:\\Users\\PS\\Desktop\\XAI code\\brain tumor\\Training'
        env = DataAugmentationEnv(datasetfolder)
        actor = build_actor(env)
        action_input, critic = build_critic(env)
        memory = SequentialMemory(limit=100000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(size=env.action_space.shape[0], theta=.15, mu=0., sigma=.3)
        agent = DDPGAgent(nb_actions=env.action_space.shape[0], actor=actor, critic=critic,
                          critic_action_input=action_input,
                          memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                          random_process=random_process, gamma=.99, target_model_update=1e-3)
        agent.compile(Adam(lr=.001, clipnorm=1.), metrics=["mae"])
        agent.fit(env, nb_steps=2000, visualize=False, verbose=1)
        sys.stdout = original_stdout

if __name__ == "__main__":
    main()