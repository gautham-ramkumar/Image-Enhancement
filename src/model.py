## CS7180 - Advanced Perception
## DATE: 09/21/2025
## Gautham Ramkumar & Sai Vamsi Rithvik Allanka

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

## U-Net based DCE-Net to predict the enhancement parameter map A and grayscale image for denoising
def build_net():
    inputs = layers.Input(shape=[None, None, 3])

    ## Encoder Layers of U-Net - First two conv layers followed by max-pooling
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    ## Second set of conv layers followed by max-pooling
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    ## Bottleneck Layer of U-Net - Two conv layers
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)

    ## U-Net Architecture works by using a decoder to reconstruct the output from the bottleneck layer
    ## We have two decoders here - one for grayscale conversion and one for enhancement parameter map A

    ## Decoder for grayscale conversion (Denoising head)
    up4_gray = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv3)
    up4_gray = layers.concatenate([up4_gray, conv2])
    conv4_gray = layers.Conv2D(64, 3, activation='relu', padding='same')(up4_gray)
    conv4_gray = layers.Conv2D(64, 3, activation='relu', padding='same')(conv4_gray)
    up5_gray = layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv4_gray)
    up5_gray = layers.concatenate([up5_gray, conv1]) # Skip connection
    conv5_gray = layers.Conv2D(32, 3, activation='relu', padding='same')(up5_gray)
    conv5_gray = layers.Conv2D(32, 3, activation='relu', padding='same')(conv5_gray)
    grayscale_output = layers.Conv2D(1, 3, activation='sigmoid', padding='same', name='grayscale_head')(conv5_gray)

    ## Decoder for image enhancement (parameter map A head)
    up4_enhance = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv3)
    up4_enhance = layers.concatenate([up4_enhance, conv2]) # Skip connection
    conv4_enhance = layers.Conv2D(64, 3, activation='relu', padding='same')(up4_enhance)
    conv4_enhance = layers.Conv2D(64, 3, activation='relu', padding='same')(conv4_enhance)
    up5_enhance = layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv4_enhance)
    up5_enhance = layers.concatenate([up5_enhance, conv1]) # Skip connection
    conv5_enhance = layers.Conv2D(32, 3, activation='relu', padding='same')(up5_enhance)
    conv5_enhance = layers.Conv2D(32, 3, activation='relu', padding='same')(conv5_enhance)
    parameter_map_A = layers.Conv2D(24, 3, activation='tanh', padding='same', name='enhancement_head')(conv5_enhance)

    return Model(inputs=inputs, outputs=[grayscale_output, parameter_map_A], name='U-Net')

## Class to build the final model that integrates both heads and performs iterative enhancement
class FinalModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.unet = build_net()
    
    ## Function to iteratively enhance the image using the predicted parameter map A
    def get_enhanced_image(self, low_light_image, parameter_map_A):
        enhanced_image = low_light_image
        A_maps = tf.split(parameter_map_A, 8, axis=3)
        for A_map in A_maps:
            enhanced_image = enhanced_image + A_map * (tf.square(enhanced_image) - enhanced_image)
        return tf.clip_by_value(enhanced_image, 0.0, 1.0)

    ## Function to call the model and get all three outputs
    def call(self, low_light_image):
        predicted_grayscale, parameter_map_A = self.unet(low_light_image)
        final_enhanced_image = self.get_enhanced_image(low_light_image, parameter_map_A)
        # Return all three for easy loss calculation
        return predicted_grayscale, parameter_map_A, final_enhanced_image

## Model summary
if __name__ == '__main__':
    model = FinalModel()
    print("Model Summary:")
    model.summary()