import tensorflow as tf
import tensorflow_datasets as tfds
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import numpy as np

def dataset():
    return tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                           with_info=True, as_supervised=True)

def model_cat_dog():
    model = tf.keras.Sequential([
        Conv2D(32, (2, 2), input_shape=(128, 128, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (2, 2), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    return model

def train_model(model):
    (train, valid, test), info = dataset()

    def pre_process_image(image, label):
        image = tf.cast(image, tf.float32)
        image = image / 255
        image = tf.image.resize(image, (128, 128))
        return image, label

    BATCH_SIZE = 32
    SHUFFLE = 1000

    train_data = train.map(pre_process_image).shuffle(SHUFFLE).repeat().batch(BATCH_SIZE)
    valid_data = valid.map(pre_process_image).repeat().batch(BATCH_SIZE)

    model.fit(train_data, steps_per_epoch=4000, epochs=2, validation_data=valid_data, validation_steps=10,
              callbacks=None)

model = model_cat_dog()
train_model(model)
model.save('h5.h5')

def predict(model, image):
    label_names = ['cat', 'dog']

    img = tf.keras.preprocessing.image.load_img(image, target_size=(128, 128))
    img_arr = np.expand_dims(img, axis=0)/255

    res = model.predict(img_arr)

    return label_names[round(res[0][0])]

model = tf.keras.models.load_model('h5.h5')

print(predict(model, 'cat.png'))
print(predict(model, 'dog1.png'))
print(predict(model, 'cat2.png'))