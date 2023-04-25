#Summary - LSTM model that goes through the citation sentiment corpus and analyses it.
#Authors: Andrew Skevington-Olivera and Marios Petrov

import DatasetModule
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense
from keras.optimizers import Adam
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#Model Architecture
model_LSTM = Sequential()
optimizer=Adam(learning_rate=1e-4)
embedding_layer=Embedding(DatasetModule.num_words,100,embeddings_initializer=tf.keras.initializers.Constant(DatasetModule.embedding_matrix))

model_LSTM.add(embedding_layer)
model_LSTM.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model_LSTM.add(Dense(1, activation='sigmoid'))
model_LSTM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model_LSTM.summary()

#Model Fitting
history=model_LSTM.fit(DatasetModule.X_train,DatasetModule.y_train,batch_size=32,epochs=6,validation_data=(DatasetModule.X_val,DatasetModule.y_val),verbose=1)

#Accuracy and Loss Figure Generation
model_metrics = pd.DataFrame(model_LSTM.history.history)
model_metrics[['loss','val_loss']].plot(ylim=[0,1])
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.show()

model_metrics[['accuracy','val_accuracy']].plot(ylim=[0,1])
plt.xlabel("Epoch")
plt.ylabel("Accuracy Value")
plt.show()




