#Und lernen Sie dann für Data3.csv (-1 / +1 2d-Punkte) einen linearen Klassifikator - (zB mit Pytorch oder Tensorflow).
import pandas as pd
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow.keras import layers


#load a dataset
data3_train = pd.read_csv('Data3.csv', sep=';', names=["x", "y", "label"])
data3_features = data3_train.copy()
data3_labels = data3_features.pop("label")
data3_labels = (data3_labels + 1)/2
data3_features = (data3_features + 2)/10
data3_features = np.array(data3_features)


#build a machine learning project
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(1)
])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


# #train and evaluate your model
model.fit(data3_features, data3_labels, epochs=1)
#
# model.evaluate(x_test,  y_test, verbose=2)
#
# probability_model = tf.keras.Sequential([
#   model,
#   tf.keras.layers.Softmax()
# ])
#
# print(probability_model(x_test[:5]))
