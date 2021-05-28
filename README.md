!pip install keras-tuner

import tensorflow as tf
from tensorflow import keras
import numpy as np


fashion_mnist=keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()

train_images=train_images/255.0
test_images=test_images/255.0

train_images[0].shape

train_images=train_images.reshape(len(train_images)28,28,1)
test_images=test_images.reshape(len(test_images)28,28,1)


def build_model(hp)
 model = keas.sequential([
   keras.layers.conv2D
       filters=hp.Int('conv_1_filter',min_value=32, max_value=128, step=16
       kernel_size=hp.choice('conv_1_kernel', values = [3,5]),
       activation='relu',
       input_shape=(28,28,1)
  ),
  keras.layers.Conv2D(
       filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
       kernel_size=hp.choice('conv_2_kernel', values=[3,5]),
       activation='relu'

  ),
  keras.layers.Flatten(),
  keras.layers.Dense(
      units=hp.Int('dense_1_units',min_value=32, max_value=128, step=16),

  ),
  keras.layers.Dense(10, activation='softmax')#output layer                   
 ])

 model.compile(optimizer=keras.optimizers.Adam(hp.choice('learning_rate',values=[1e-2, 1e-3])),
             loss='sparse_categorial_crossentropy' ,
             metrics=['accuracy'])
 return model

from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

tuner_search=RandomSearch(build_model,
                          objectives='val_accuracy',
                          max_trials=5,directory='output',project_name="Mnist Fashion")

tuner_search.search(train_images,train_labels,epochs=3,validation_split=0.1)

model=tunner search.get_best_models(num_models=1)[0]

model.summary()
