import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:-1]
y = dataset.iloc[:, -1]

X = pd.get_dummies(X, columns=['Geography', 'Gender'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history=ann.fit(X_train, y_train, batch_size=32, epochs=120)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
sns.lineplot(x=range(1, 121), y=history.history['accuracy'])
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
sns.lineplot(x=range(1, 121), y=history.history['loss'])
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.show()


import pickle

ann.save('churn_model.h5')

with open('scaler.pkl', 'wb') as f:
    pickle.dump(sc, f)

# Save the dummy columns list to align input later
dummy_columns = X.columns.tolist()
with open('dummy_columns.pkl', 'wb') as f:
    pickle.dump(dummy_columns, f)

print("Model, scaler, and dummy columns saved!")
