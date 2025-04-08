from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os

def load_unlabeled_dataset(path, image_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        path,
        target_size=image_size,
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        color_mode='rgb'
    )
    return generator

def simclr_augment(image):
    image = tf.image.resize(image, [224, 224])
    if tf.shape(image)[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.cast(image, tf.float32)
    return image

def generate_pair(batch):
    batch = tf.convert_to_tensor(batch, dtype=tf.float32)
    augmented_1 = tf.map_fn(simclr_augment, batch)
    augmented_2 = tf.map_fn(simclr_augment, batch)
    return (augmented_1, augmented_2)