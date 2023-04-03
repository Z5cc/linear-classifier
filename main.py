import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import seaborn as sns
import matplotlib.pyplot as plt


# load training data
train_data = pd.read_csv('Data3.csv', sep=';', names=["x", "y", "label"])
train_features = train_data.copy()
train_labels = train_features.pop("label")
train_labels = (train_labels + 1)/2
train_features = (train_features + 2)/10
train_features = np.array(train_features)

# load testing data
test_data = pd.read_csv('Data3.csv', sep=';', names=["x", "y", "label"])
test_features = test_data.copy()
test_labels = test_features.pop("label")
test_labels = (test_labels + 1)/2
test_features = (test_features + 2)/10
test_features = np.array(test_features)




# build a machine learning project
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])


# train and evaluate your model
model.fit(train_features, train_labels, epochs=3)
#
model.evaluate(x_test,  y_test, verbose=2)
#
# probability_model = tf.keras.Sequential([
#   model,
#   tf.keras.layers.Softmax()
# ])
#
# print(probability_model(x_test[:5]))


# # plot
# sns.scatterplot(data=train_data, x='x', y='y', hue='label', style='predicted_label')
#
# # TODO: get weights from model
# boundary_x = np.linspace(np.min(train_data.x), np.max(train_data.x), 100)
# boundary_y = -(weights[0] * boundary_x + bias) / weights[1]
# sns.lineplot(x=boundary_x, y=boundary_y)
#
# plt.show()
