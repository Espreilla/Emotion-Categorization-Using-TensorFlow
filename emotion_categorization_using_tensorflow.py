# -*- coding: utf-8 -*-
"""
## Emotion Categorization Using TensorFlow

# Import Library untuk proyek
"""

from google.colab import drive

#untuk dataframe
import pandas as pd
import re

#Import sklearn untuk preprocessing dan plit data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#Import tensorflow
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
print(tf.__version__)

#Import matplotlib untuk visualisasi data
import matplotlib.pyplot as plt

"""# Mengimport Dataset dan Melakukan Preprocessing Data

Dataset yang digunakan merupakan "Emotions Dataset for NLP" -> Train.txt [link text](https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp)

Data yang diambil hanya data train.txt karena akan dilakukan simulasi pembagian antara train set dan validation set yaitu sebesar 80% untuk train set dan 20% untuk validation set.

Data terbagi atas 2 columns yaitu kolom "Document" dan kolom "Emotion". Model dibuat untuk mengembangkan emotion classification menggunakan NLP. Selanjutnya model ini dapat digunakan untuk menjawab pertanyaan seperti "sentimen apa yang didapatkan berdasarkan komentar pelanggan?".
"""

#mount drive
drive.mount('/content/drive')

"""Dilakukan import dataset dan penambahan label untuk header yaitu "Document" dan "Emotion""""

#importing data
data = pd.read_csv("/content/drive/My Drive/Dicoding/Submission1/train.txt", sep = ";", names = ["Document", "Emotion"])
data.head()

"""Mengecek klasifikasi data dan jumlahnya."""

data["Emotion"].value_counts()

"""Melakukan preprocessing data yaitu penghapusan special character dan "Document" sebelumnya."""

# Menghapus special character di kolom text
data["Doc"] = data["Document"].map(lambda x: re.sub(r'\W+', ' ', x))
# drop kolom document lama
data = data.drop(["Document"], axis=1)
data.head()

"""Melakukan pengecekan nilai kosong / NaN"""

data.isnull().values.any()

"""Diketahui bahwa nilai kosongnya "False" yang artinya tidak ada nilai kosong pada data.

Selanjutnya, akan dilakukan pelabelan pada Emotion.
"""

category = pd.get_dummies(data.Emotion)
data_baru = pd.concat([data, category], axis=1)
data_baru = data_baru.drop(columns="Emotion")
data_baru

"""Agar dapat diproses oleh model, nilai-nilai dari dataframe diubah ke dalam tipe data numpy array menggunakan atribut values."""

text = data_baru["Doc"].values
label = data_baru[["anger", "fear", "joy", "love", "sadness", "surprise"]].values

"""# Membagi data menjadi data training dan data validation

Membagi data sebagai train set sebanyak 80% dan validation set sebanyak 20%.
"""

text_train, text_val, label_train, label_val = train_test_split(text, label, test_size = 0.2)

"""# Menggunakan fungsi tokenizer"""

tokenizer = Tokenizer(num_words=5000, oov_token='x')
tokenizer.fit_on_texts(text_train)
tokenizer.fit_on_texts(text_val)
 
seq_train = tokenizer.texts_to_sequences(text_train)
seq_val = tokenizer.texts_to_sequences(text_val)
 
padded_train = pad_sequences(seq_train)
padded_val = pad_sequences(seq_val)

"""# Model Menggunakan LSTM dan Sequential"""

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=16),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

"""Model compile dengan optimizer Adam."""

Adam(learning_rate=0.00146, name='Adam')
model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

"""Melakukan implementasi callback untuk mendapatkan akurasi > 90%"""

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.9 and logs.get('val_accuracy')>0.9):
      print("\nAkurasi train dan validasi telah mencapai nilai > 90%!")
      self.model.stop_training = True
callbacks = myCallback()

"""Selanjutnya, melatih model kita dengan memanggil fungsi fit()."""

num_epochs = 30
history = model.fit(padded_train, label_train, epochs=num_epochs, validation_data=(padded_val, label_val), verbose=1, callbacks=[callbacks])

"""# Mengeluarkan Plot Accuracy dan Plot Loss untuk Train Set dan Validation Set"""

# Plot Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy Model')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Plot Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()