#Summary - LSTM+CNN model that goes through the citation sentiment corpus and analyses it.
#Authors: Marios Petrov and Andrew Skevington-Olivera

import DatasetModule
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Embedding
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#Model Architecture
model_CNN=Sequential()
embedding_layer=Embedding(DatasetModule.num_words,100,embeddings_initializer=tf.keras.initializers.Constant(DatasetModule.embedding_matrix))
optimizer=Adam(learning_rate=1e-4)

model_CNN.add(embedding_layer)
model_CNN.add(Dropout(0.2))
model_CNN.add(Conv1D(64, 5, activation='relu'))
model_CNN.add(MaxPooling1D(pool_size=4))
model_CNN.add(LSTM(100))
model_CNN.add(Dense(3,activation='sigmoid'))
model_CNN.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model_CNN.summary()

#Model Fitting
history=model_CNN.fit(DatasetModule.X_train,DatasetModule.y_train,batch_size=32,epochs=2,validation_data=(DatasetModule.X_val,DatasetModule.y_val),verbose=1)

model_loss = pd.DataFrame(model_CNN.history.history)

#Attempt at implementing a confusion matrix to generate precision, recall, and Fscores by hand since the keras metrics weren't working
#yhat = model_CNN.predict(DatasetModule.X_train)
#ytrue = DatasetModule.y_train.astype(int).tolist()
#yhat = yhat.astype(int).tolist()
#print(confusion_matrix(ytrue,yhat))

#Accuracy and Loss Figure Generation
model_metrics = pd.DataFrame(model_CNN.history.history)
model_metrics[['loss','val_loss']].plot(ylim=[0,1])
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.show()

model_metrics[['accuracy','val_accuracy']].plot(ylim=[0,1])
plt.xlabel("Epoch")
plt.ylabel("Accuracy Value")
plt.show()



