import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt


# load data
train_data = pd.read_csv('Data3.csv', sep=';', names=["x", "y", "label"])
test_data = pd.read_csv('Data3_test.csv', sep=';', names=["x", "y", "label"])
# normalize data
data = train_data.append(test_data)
data = (data-data.min())/(data.max()-data.min())
train_data = data.iloc[:100]
test_data = data.iloc[100:]
# split into labels and features
train_features = train_data.copy()
train_labels = train_features.pop("label")
train_features = np.array(train_features)
test_features = test_data.copy()
test_labels = test_features.pop("label")
test_features = np.array(test_features)


# build model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])


# train and evaluate model
model.fit(train_features, train_labels, epochs=1000)
[[[w0], [w1]], [b]] = model.layers[0].get_weights()

model.evaluate(test_features, test_labels, verbose=2)


# plot
sns.scatterplot(data=test_data, x='x', y='y', hue='label')

boundary_x = np.linspace(np.min(data.x), np.max(data.x), 100)
boundary_y = (0.5/w1) - (b/w1) - (w0/w1)*boundary_x
sns.lineplot(x=boundary_x, y=boundary_y)

plt.show()
