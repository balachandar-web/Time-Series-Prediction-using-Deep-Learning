import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint

X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')

look_back = 12
model = create_model(look_back)

checkpoint = ModelCheckpoint('model/saved_model/model.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X_train, Y_train, batch_size=1, epochs=20, callbacks=callbacks_list)

