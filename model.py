import numpy as np
import pandas as pd

from PIL import Image

PADDING_SIZE = 20
IMAGE_SIZE = 256
BATCH_SIZE = 4

ALL_WORD = ['<none>', '', 'statement', 'if', 'else', 'elseif', 'while', 'end', '<START>', '<END>']
N_ONEHOT_WORD = len(ALL_WORD)
CHARS_MAP = {v: k for k, v in enumerate(ALL_WORD)}

from keras.layers import Input, Dense, Flatten, Concatenate, Conv2D, MaxPooling2D, LSTM, RepeatVector, Embedding
from keras.layers import Reshape, UpSampling2D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import tensorflow as tf

def generate_model_small():
  word_input = Input(batch_shape=(BATCH_SIZE, PADDING_SIZE))
  image_input = Input(batch_shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
  
  img = Conv2D(16, (3, 3), padding='same', activation='relu')(image_input)

  img = MaxPooling2D()(img)
  img = MaxPooling2D()(img)
  img = MaxPooling2D()(img)
  img = MaxPooling2D()(img)
  img = MaxPooling2D()(img)

  img = Flatten()(img)
  img = Dense(64)(img)
  img = RepeatVector(PADDING_SIZE)(img)

  w = Embedding(N_ONEHOT_WORD, 64)(word_input)

  x = Concatenate()([w, img])
  x = LSTM(32)(x)

  y = Dense(32)(x)
  y = Dense(N_ONEHOT_WORD, activation='softmax', name='y_word')(y)

  y_pos = Dense(32)(x)
  y_pos = Dense(2, name='y_position')(y_pos)

  optimizer = Adam(lr=0.0001)
  model = Model(inputs=[word_input, image_input], outputs=[y_pos, y])
  
  model.compile(optimizer=optimizer,
                loss=['mse', 'categorical_crossentropy'],
                metrics={'y_position':'mse', 'y_word':'accuracy'})
  
  return model

def generate_model():
  word_input = Input(batch_shape=(BATCH_SIZE, PADDING_SIZE))
  image_input = Input(batch_shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
  
  img = Conv2D(32, (3, 3), padding='same', activation='relu')(image_input)
  
  img = MaxPooling2D()(img)
  img = Conv2D(64, (3,3), padding='same', activation='relu')(img)
  
  img = MaxPooling2D()(img)
  img = Conv2D(92, (3,3), padding='same', activation='relu')(img)
  
  img = MaxPooling2D()(img)
  img = Conv2D(128, (3,3), padding='same', activation='relu')(img)
  
  img = MaxPooling2D()(img)
  img = Conv2D(192, (3,3), padding='same', activation='relu')(img)
  
  img = MaxPooling2D()(img)
  img = Conv2D(224, (3,3), padding='same', activation='relu')(img)
  
  img = Flatten()(img)
  img = Dense(2048)(img)
  img = RepeatVector(PADDING_SIZE)(img)
  
  w = Embedding(N_ONEHOT_WORD, 64)(word_input)
  w = LSTM(512, return_sequences=True)(w)
  w = LSTM(512, return_sequences=True)(w)
  
  x = Concatenate()([w, img])
  x = LSTM(512, return_sequences=True)(x)
  x = LSTM(512)(x)
  
  y = Dense(512)(x)
  y = Dense(N_ONEHOT_WORD, activation='softmax', name='y_word')(y)
  
  y_pos = Dense(512)(x)
  y_pos = Dense(2, name='y_position')(y_pos)

  optimizer = Adam(lr=0.0001)
  model = Model(inputs=[word_input, image_input], outputs=[y_pos, y])
  
  model.compile(optimizer=optimizer,
                loss=['mse', 'categorical_crossentropy'],
                metrics={'y_position':'mse', 'y_word':'accuracy'})

model = None

def convert_to_feature_list(feature_words):
  onehots = []
  for word in feature_words:
    onehots.append(CHARS_MAP[word])
  return np.array(onehots)

def read_image(image_file):
  image = Image.open(image_file).convert('L')
  width, height = image.size
  image = image.convert()
  image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
  image = np.asarray(image).reshape((IMAGE_SIZE, IMAGE_SIZE, 1))
  return 1 - (image / 255), width, height

def flow2code(image_path):
  global model

  if model is None:
    print("Reload model !!!")
    model = generate_model_small()
    model.load_weights("./model/model-weight.hdf5")
  
  image, img_width, img_height = read_image(image_path)

  tokens = np.zeros((BATCH_SIZE, PADDING_SIZE))
  images = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
  results = []
  position = []
  
  token = [''] * (PADDING_SIZE + 1)
  
  for i in range(BATCH_SIZE):
    images[i] = image
  
  for i in range(20):
    token = token[1:]
    
    tokens[BATCH_SIZE - 1] = convert_to_feature_list(token)
    
    pos, predict = model.predict([tokens, images])
    predict_token = ALL_WORD[np.argmax(predict[BATCH_SIZE - 1])]
    results.append(predict_token)
    
    token += [predict_token]
    position += [pos[BATCH_SIZE - 1]]
    
    if predict_token == "<END>":
        break

  position = np.array(position)
  position[position[:,0] > 0,0] *= img_width / 10
  position[position[:,1] > 0,1] *= img_height / 10

  return {
    "tokens": results,
    "positions": position,
    "img_width": img_width,
    "img_height": img_height
  }