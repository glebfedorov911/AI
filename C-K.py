import os # from Celcium to Kelvin
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense

c = np.array([0, 70, 18, -18, -273, 85, -85, 27])
k = np.array([273, 343, 291, 255, 0, 358, 188, 300])

model = keras.Sequential()
model.add(Dense(units=1, input_shape=(1, ), activation='linear'))

model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(0.1))

history = model.fit(c, k, epochs=6000, verbose=0)

plt.plot(history.history['loss'])
plt.grid(True)
plt.show()

print(model.predict([100, 0, 17]))
print(model.get_weights())
