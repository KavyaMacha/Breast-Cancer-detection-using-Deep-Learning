#!/usr/bin/env python
# coding: utf-8

# In[2]:


from platform import python_version

print(python_version())


# In[3]:


pip install tensorflow


# # breast cancer using ultrasound

# In[4]:


#breast cancer using ultrasound
import cv2
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import os

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
get_ipython().run_line_magic('matplotlib', 'inline')

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from random import randint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


print(tf.__version__)


# In[5]:


BATCH_SIZE = 32
EPOCHS = 10

IMAGE_SIZE = (150, 150)

tf.random.set_seed(0)


# In[6]:


device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))


# In[7]:


pip install opendatasets


# In[8]:


import opendatasets as od


# In[9]:


for dirname, _, filenames in os.walk('Dataset_BUSI_with_GT'):
    print(dirname)


# # Define classes

# In[10]:


CLASSES = {'benign': 0, 'malignant': 1, 'normal': 2}


# # Get Train Sample

# In[11]:


def shuffle_prune(df, BATCH_SIZE):
    df = shuffle(df, random_state=42)
    df.reset_index(drop=True, inplace=True)
    df = df[ : df.shape[0] // BATCH_SIZE * BATCH_SIZE]
    return df


# In[12]:


filenames = tf.io.gfile.glob('Documents/Dataset_BUSI_with_GT/Seg_train_only_images/Seg_train_only_images/*/*')
image_path_df_train = pd.DataFrame(data={'filename': filenames, 'class': [x.split('\\')[-2] for x in filenames]})
image_path_df_train = shuffle_prune(image_path_df_train, BATCH_SIZE)
image_path_df_train['class'] = image_path_df_train['class'].map(CLASSES)

print('Train sample: ', len(image_path_df_train['class']), dict(image_path_df_train['class'].value_counts()))


# # Get Test Sample

# In[13]:


filenames = tf.io.gfile.glob('Documents\Dataset_BUSI_with_GT\Seg_test_only_images\Seg_test_only_images/*/*')
image_path_df_test = pd.DataFrame(data={'filename': filenames, 'class': [x.split('\\')[-2] for x in filenames]})

print('Test sample: ', len(image_path_df_test['class']), dict(image_path_df_test['class'].value_counts()))


# # Get Validation sample from test sample

# In[14]:


image_path_df_test, image_path_df_val  = train_test_split(image_path_df_test, test_size=0.5, random_state=42, stratify=image_path_df_test['class'])
image_path_df_test = shuffle_prune(image_path_df_test, BATCH_SIZE)
image_path_df_test['class'] = image_path_df_test['class'].map(CLASSES)

image_path_df_val = shuffle_prune(image_path_df_val, BATCH_SIZE)
image_path_df_val['class'] = image_path_df_val['class'].map(CLASSES)

print('Test sample: ', len(image_path_df_test['class']), dict(image_path_df_test['class'].value_counts()))
print('Val  sample: ', len(image_path_df_val['class']), dict(image_path_df_val['class'].value_counts()))


# # Get files for prediction

# In[15]:


filenames = tf.io.gfile.glob('Documents/Dataset_BUSI_with_GT/Seg_pred2/Seg_pred2/*')

image_path_df_predict = pd.DataFrame(data={'filename': filenames, 'class': np.nan})
print(f'Number filenames: {len(image_path_df_predict)}')


# # Get arrays and labels

# In[16]:


def get_images_and_labels_arrays(df):
    images = []
    for file in df['filename']:
        image = cv2.imread(file)
        image = cv2.resize(image,IMAGE_SIZE)
        images.append(image)
    images = np.array(images)
    
    labels = df.loc[:, 'class']
    return images, labels


# In[17]:


train_images, train_labels = get_images_and_labels_arrays(image_path_df_train)

print(f'Shape of train set: {train_images.shape}')
print(f'Shape of train set: {train_labels.shape}')


# In[18]:


val_images, val_labels = get_images_and_labels_arrays(image_path_df_val)

print(f'Shape of validation set: {val_images.shape}')
print(f'Shape of validation set: {val_labels.shape}')


# In[19]:


test_images, test_labels = get_images_and_labels_arrays(image_path_df_test)

print(f'Shape of test set: {test_images.shape}')
print(f'Shape of test set: {test_labels.shape}')


# # we have  3 classes in this work like this.

# In[20]:


f,ax = plt.subplots(3,3) 
f.subplots_adjust(0,0,3,3)
for i in range(0,3,1):
    for j in range(0,3,1):
        rnd_number = randint(0,len(train_images))
        ax[i,j].imshow(train_images[rnd_number])
        ax[i,j].set_title([key for key, val in CLASSES.items() if val == train_labels[rnd_number]][0])
        ax[i,j].axis('off')


# # Define CNN Keras model and Compile

# In[21]:


def create_model():
    
    with tf.device('/gpu:0'):
    
        input_layer = layers.Input(shape=(*IMAGE_SIZE, 3), name='input') 
        x = layers.BatchNormalization()(input_layer)

        x = layers.Conv2D(filters=64, kernel_size=3, 
                          activation='relu', padding='same', 
                          name='conv2d_1')(x)
        x = layers.MaxPool2D(pool_size=2, name='maxpool2d_1')(x)
        x = layers.Dropout(0.1, name='dropout_1')(x)

        x = layers.Conv2D(filters=128, kernel_size=3, 
                          activation='relu', padding='same', 
                          name='conv2d_2')(x)
        x = layers.MaxPool2D(pool_size=2, name='maxpool2d_2')(x)
        x = layers.Dropout(0.1, name='dropout_2')(x)

        x = layers.Conv2D(filters=256, kernel_size=3, 
                          activation='relu', padding='same', 
                          name='conv2d_3')(x)
        x = layers.MaxPool2D(pool_size=2, name='maxpool2d_3')(x)
        x = layers.Dropout(0.1, name='dropout_3')(x)

        x = layers.Conv2D(filters=512, kernel_size=3, 
                          activation='relu', padding='same', 
                          name='conv2d_4')(x)
        x = layers.MaxPool2D(pool_size=2, name='maxpool2d_4')(x)
        x = layers.Dropout(0.1, name='dropout_4')(x)

        x = layers.Conv2D(filters=1024, kernel_size=3, 
                          activation='relu', padding='same', 
                          name='conv2d_5')(x)
        x = layers.MaxPool2D(pool_size=2, name='maxpool2d_5')(x)
        x = layers.Dropout(0.1, name='dropout_5')(x)
        

        x = layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)
        x = layers.BatchNormalization()(x)
       
        x = layers.Dense(128,activation='relu')(x)
        
        output = layers.Dense(units=len(CLASSES), 
                              activation='softmax', 
                              name='output')(x)
        model = Model (input_layer, output)    
        model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])

    return model

model = create_model()
model.summary()


# # Run model Training

# In[22]:


init_time = datetime.datetime.now()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 1, verbose=1, factor=0.3, min_lr=0.000001)

trained = model.fit(
                    train_images, train_labels,
                    validation_data = (val_images, val_labels),
                    batch_size = BATCH_SIZE, 
                    epochs=EPOCHS,
                    callbacks=[learning_rate_reduction],
    )

requared_time = datetime.datetime.now() - init_time
print(f'\nRequired time:  {str(requared_time)}\n')


# # Training Process

# In[23]:


plt.plot(trained.history['accuracy'])
plt.plot(trained.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(trained.history['loss'])
plt.plot(trained.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# # Evalute the trained model

# In[24]:


test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print('\naccuracy:', test_acc, '  loss: ',test_loss)


# In[ ]:





# # Prediction

# In[25]:


predict = np.argmax(model.predict(test_images), axis=1)
predict


# # Classification and confusion matrix
# 

# In[26]:


print(classification_report(test_labels, predict), '\n')
cm = confusion_matrix(test_labels, predict)
sns.heatmap(cm, annot=True, cmap="Blues", fmt='.0f', cbar=False)
plt.show()


# # Let's try pretrained model for example VGG19

# In[27]:


def create_model():
    with tf.device('/gpu:0'):
        pretrained_model = tf.keras.applications.VGG19(
            weights='imagenet',
            include_top=False ,
            input_shape=[*IMAGE_SIZE, 3]
        )
        pretrained_model.trainable = False

        
        
        input_layer = layers.Input(shape=(*IMAGE_SIZE, 3), name='input') 
        
        x = pretrained_model(input_layer)

        x = layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)
        x = layers.BatchNormalization()(x)       
        x = layers.Dense(128,activation='relu')(x)
        
        output = layers.Dense(units=len(CLASSES), 
                              activation='softmax', 
                              name='output')(x)


        model = Model (input_layer, output)    
        model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])

        return model

model = create_model()
model.summary()


# In[28]:


init_time = datetime.datetime.now()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 1, verbose=1, factor=0.3, min_lr=0.000001)

trained = model.fit(
                    train_images, train_labels,
                    validation_data = (val_images, val_labels),
                    batch_size = BATCH_SIZE, 
                    epochs=EPOCHS,
                    callbacks=[learning_rate_reduction],
    )

requared_time = datetime.datetime.now() - init_time
print(f'\nRequired time:  {str(requared_time)}\n')


# In[29]:


plt.plot(trained.history['accuracy'])
plt.plot(trained.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(trained.history['loss'])
plt.plot(trained.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# # Prediction
# 

# In[30]:


predict = np.argmax(model.predict(test_images), axis=1)
predict


# # Classification report & confusion matrix

# In[31]:


print(classification_report(test_labels, predict), '\n')
cm = confusion_matrix(test_labels, predict)
sns.heatmap(cm, annot=True, cmap="Blues", fmt='.0f', cbar=False)
plt.show()


# # Let's predict the unlabeled data
# 

# In[32]:


to_predict_images, to_predict_labels = get_images_and_labels_arrays(image_path_df_predict)
print(f'Shape of images set to prediction: {to_predict_images.shape}')


# In[33]:


predict = np.argmax(model.predict(to_predict_images), axis=1)
predict


# # Let's check random images

# In[34]:


f,ax = plt.subplots(5,5) 
f.subplots_adjust(0,0,3,3)
for i in range(0,5,1):
    for j in range(0,5,1):
        rnd_number = randint(0,len(predict))
        ax[i,j].imshow(to_predict_images[rnd_number])
        ax[i,j].set_title([key for key, val in CLASSES.items() if val == predict[rnd_number]][0])
        ax[i,j].axis('off')


# # Actual label vs Predicted label

# In[35]:


import numpy as np
import tensorflow as tf

# Load your trained model
#model = tf.keras.models.load_model('model.h5')

# Load the test images and labels
#test_images = ...  your test images
#test_labels = ... # the actual labels for the test images
train_images, train_labels = get_images_and_labels_arrays(image_path_df_train)

# Make predictions for the test images
predictions = model.predict(test_images)

# Convert the predictions to the class labels
to_predict_labels = np.argmax(predictions, axis=1)

# Compare the actual and predicted labels
for i in range(len(test_images)):
    print("Image {}: Actual label = {}, Predicted label = {}".format(i, test_labels[i],to_predict_labels[i]))


# In[ ]:




