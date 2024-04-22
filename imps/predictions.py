from tensorflow.keras.models import load_model  
from keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

'''unused imports'''
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn import metrics
# import tensorflow as tf
# import os
# import numpy as np
# import dlib
# import cv2
# import matplotlib.pyplot as plt
# import addons as tfa
# from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.utils import image_dataset_from_directory
# from tensorflow import load_model
# import tensorflow_addons as tfa


'''my own imports, was not present in the official code'''
from os import listdir


def predict_start(dataset:str, model):
    folders = listdir(r"Datasets")
    if not (dataset in folders):
        return {"Error": "No Dataset with this name exists.", "Completion": "None"}
    if not ("TEST" in folders):
        return {"Error": "No Test dataset detected.", "Completion": "None"}

    # Check the specific Model's availability
    folders = listdir(r"Model")
    if not (model in folders):
        return {"Error": "No such model file exist, check the model's name or path.", "Completeion": "None"}

    dataset = rf"{dataset}"
    img_rows, img_cols =224, 224
    input_shape = (img_rows,img_cols,3)
    batch_size = 16
    # for the provided dataset
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    dataset_generator = test_datagen.flow_from_directory(
        rf"Datasets\{dataset}",
        # subset='validation',
        target_size=(img_rows,img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        classes=['fake', 'real']
        )
    # for the test dataset
    test_generator = test_datagen.flow_from_directory(
        r"Datasets\TEST",
        # subset='validation',
        target_size=(img_rows,img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        classes=['fake', 'real']
        )
    dataset_samples = len(dataset_generator.filenames)
    test_samples = len(test_generator.filenames)
    Y_dataset = dataset_generator.classes
    Y_test = test_generator.classes
    model = load_model(rf'Model\\{model}')

    # y_pred_test = model.predict_generator(test_generator, test_samples/batch_size, workers=1)
    # y_pred_dataset = model.predict_generator(dataset_generator, dataset_samples/batch_size, workers=1)
    '''
    The below statements uses 'model.predict', instead of 'model.predict_generator' . Because the later is a depricated method
    and there is no difference in accuracy, but if you need 'model.prredict_generator', comment two statements below and un-comment
    two statements above this comment.'''

    
    y_pred_test = model.predict(test_generator, test_samples/batch_size, workers=1)
    y_pred_dataset = model.predict(dataset_generator, dataset_samples/batch_size, workers=1)
    
    # accuracy calculations
    accuracy_test = accuracy_score(Y_test, y_pred_test.argmax(axis=-1))
    accuracy_dataset = accuracy_score(Y_dataset, y_pred_dataset.argmax(axis=-1))

    '''This ASR is calculated by non-percentage values, meaning the accuracies are not multiplied by 100'''
    asr = (accuracy_test - accuracy_dataset) / accuracy_test
    # '''for ASR in precentage, comment the below line if not needed in percent'''
    asr_per = asr * 100
    return {"Accuracy before Attack": accuracy_test, "Accuracy after Attack": accuracy_dataset, "ASR": asr, "ASR Percent": asr_per}
    accuracy_test_percent = accuracy_test * 100
    accuracy_dataset_percent = accuracy_dataset * 100

