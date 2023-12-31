from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_test_data(test_folder, batch_size=16):
    """
    Load test data from a specified folder.
    """
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_folder,
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    return test_generator

def plot_confusion_matrix(cm, class_labels):
    """
    Plot a beautiful confusion matrix with larger fonts.
    """
    sns.set(context='talk', style='whitegrid', palette='deep', font='sans-serif', font_scale=1.2)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted labels', fontsize=18)
    plt.ylabel('True labels', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()


def evaluate_model(model_path, test_folder):
    """
    Load a model and evaluate it on the test set.
    """
    # Load the model
    model = load_model(model_path)

    # Load test data
    test_generator = load_test_data(test_folder)

    # Get the number of samples and number of classes
    num_samples = test_generator.samples
    num_classes = test_generator.num_classes

    # Predict the whole test set
    test_generator.reset()
    predictions = model.predict(test_generator, steps=np.ceil(num_samples / test_generator.batch_size), verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Ensure the number of class labels matches the number of classes predicted by the model
    if len(class_labels) != num_classes:
        raise ValueError(f"Number of class labels ({len(class_labels)}) does not match number of classes predicted by the model ({num_classes}).")

    # Compute confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    print('Confusion Matrix:')
    print(cm)

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_labels)

    # Compute classification report
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print('Classification Report:')
    print(report)


# 使用测试集路径调用 evaluate_model 函数
test_folder ="C:\\Users\\DELL\\Desktop\\code\\dataset\\covid-xray\\Data\\train"# 替换为测试集文件夹的路径
model_path = "C:\\Users\\DELL\\Desktop\\data processing\\covid\\Densenet121\\COVID DenseNet121.h5"# 模型文件路径
evaluate_model(model_path, test_folder)
