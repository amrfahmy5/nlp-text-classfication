import tensorflow as tf
from Embedding import Embedding as dm
from keras.layers import LSTM, Activation, Dense, Dropout, Input , Embedding
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import to_categorical
from keras.optimizers import adam


embedder = dm()
embedder.build_model()
X_train, X_test, y_train, y_test = embedder.spliting_data()

embedding_dim = embedder.n
vocab_size = 4000

max_words = 1000
max_len = 200

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    # tf.keras.layers.Dense(vocab_size),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.summary()

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])
num_epochs = 5
history = model.fit(X_train,y_train, epochs=num_epochs,
                    validation_data=(X_test,y_test), verbose=1)
import matplotlib.pyplot as plt

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Embedding(max_words,embedding_dim,input_length= max_len))
# model.add(tf.keras.layers.SpatialDropout1D(0.2))
# model.add(tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))
# model.add(tf.keras.layers.Dense(3, activation='softmax'))


# #opt = SGD(lr =0.01)
# #opt = adam(lr=0.001, decay=1e-6)
# model.compile(loss='sparse_categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

# epochs = 5
# batch_size = 64

# history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)] )

# history = model.fit(X_train,y_train, epochs=epochs,
#                     validation_data=(X_test,y_test), verbose=1)




#max_words = 1000
#max_len = 200
#num_epochs =5
#
#
#def RNN():
#    inputs = Input(name='inputs',shape=[max_len])
#    layer = Embedding(max_words,50,input_length=max_len)(inputs)
#    layer = LSTM(64)(layer)
#    layer = Dense(256,name='FC1')(layer)
#    layer = Activation('relu')(layer)
#    layer = Dropout(0.5)(layer)
#    layer = Dense(1,name='out_layer')(layer)
#    layer = Activation('softmax')(layer)
#    model = Model(inputs=inputs,outputs=layer)
#    return model
#
#model = RNN()
#model.summary()
#model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
#model.fit(X_train,y_train, epochs=num_epochs,
#                   validation_data=(X_test,y_test), verbose=1)









def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
#plot_graphs(history, "accuracy")
#plot_graphs(history, "loss")
# labels = {
#     0 : "Negative",
#     1 : "Normal",
#     2: "Positive"
#         }
# txt = input("Tweet Now ...")
# tokens  = parser.findall(txt)
# avg = get_average(tokens)
# print(labels[classifier.predict([avg])[0]])