from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model

def build_simclr_model(input_shape=(224, 224, 3), projection_dim=128):
    base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape, pooling='avg')
    inputs = layers.Input(shape=input_shape)
    features = base_model(inputs)
    x = layers.Dense(256, activation='relu')(features)
    x = layers.Dense(projection_dim)(x)
    model = Model(inputs=inputs, outputs=x)
    return model