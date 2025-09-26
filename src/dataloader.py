## CS7180 - Advanced Perception - Final Dataloader Script
## DATE: 09/21/2025
## Gautham Ramkumar & Sai Vamsi Rithvik Allanka

import tensorflow as tf
from glob import glob

IMAGE_SIZE = 256
BATCH_SIZE = 16

## LOL-V1 Dataset Paths
our_low = "/home/gautham/Documents/Advanced_Perception/Project1/Dataset/LOLdataset/our485/low/*"
our_high = "/home/gautham/Documents/Advanced_Perception/Project1/Dataset/LOLdataset/our485/high/*"

## EVAL15 Dataset Paths
eval_low = "/home/gautham/Documents/Advanced_Perception/Project1/Dataset/LOLdataset/eval15/low/*"
eval_high = "/home/gautham/Documents/Advanced_Perception/Project1/Dataset/LOLdataset/eval15/high/*"

## Load and preprocess images
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    return image / 255.0

## Load paired low-light and high-light images
def load_paired_data(low_path, high_path):
    low_image = load_image(low_path)
    high_image = load_image(high_path)
    return low_image, high_image

## Create training dataset and pair low and high light images
def get_train_dataset():
    print("Loading paired training data")
    train_low = sorted(glob(our_low))
    train_high = sorted(glob(our_high))
    
    dataset = tf.data.Dataset.from_tensor_slices((train_low, train_high))
    dataset = dataset.map(load_paired_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    print(f"Training dataset created with {len(train_low)} paired images.")
    return dataset

## Create validation dataset and pair low and high light images
def get_val_dataset():
    print("Loading paired validation data")
    val_low = sorted(glob(eval_low))
    val_high = sorted(glob(eval_high))

    dataset = tf.data.Dataset.from_tensor_slices((val_low, val_high))
    dataset = dataset.map(load_paired_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    print(f"Validation dataset created with {len(val_low)} paired images.")
    return dataset