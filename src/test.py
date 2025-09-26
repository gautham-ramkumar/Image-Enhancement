## CS7180 - Advanced Perception
## DATE: 09/21/2025
## Gautham Ramkumar & Sai Vamsi Rithvik Allanka

import os
import numpy as np
import keras
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from glob import glob

from train import SupervisedTrainer

## Load the best model weights saved during training
model = SupervisedTrainer()
model(tf.random.normal([1, 256, 256, 3]))
# Load the weights that were saved by the ModelCheckpoint callback
model.load_weights("/home/gautham/Documents/Advanced_Perception/Project1/final_supervised_model.weights.h5")
print("Best model weights loaded successfully.")

## Define a function to handle inference and visualization for a single image
def plot_results(images, titles, figure_size=(20, 12)):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        _ = plt.imshow(images[i])
        plt.axis("off")
    plt.show()

## Define the inference function for a single image
def infer(original_image):
    image = keras.utils.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    
    ## The model returns a tuple (enhanced_image, parameter_map_A, grayscale_image)
    _, _, enhanced_image_tensor = model(image)
    
    ## Process the tensor to be a viewable image
    enhanced_image_tensor = tf.cast((enhanced_image_tensor[0] * 255), dtype=np.uint8)
    output_image = Image.fromarray(enhanced_image_tensor.numpy())
    return output_image

## Test Dataset Paths
test_low = sorted(glob("/home/gautham/Documents/Advanced_Perception/Project1/Dataset/LOLdataset/eval15/low/*"))
test_high = sorted(glob("/home/gautham/Documents/Advanced_Perception/Project1/Dataset/LOLdataset/eval15/high/*"))

## Create a directory to save the results
output_dir = "test_outputs"
os.makedirs(output_dir, exist_ok=True)

## Run inference on the test dataset and save results
for low_path, high_path in zip(test_low, test_high):
    original_image = Image.open(low_path)
    ground_truth_image = Image.open(high_path)
    
    ## Get the enhanced image from your model
    enhanced_image = infer(original_image)
    
    ## Save the enhanced image to the output folder
    image_name = os.path.basename(low_path)
    enhanced_image.save(os.path.join(output_dir, image_name))
    
    ## Plot the original, ground truth, and your model's output for comparison
    plot_results(
        [original_image, ground_truth_image, enhanced_image],
        ["Original Low-Light", "Ground Truth", "Enhanced by Your Model"],
    )

print(f"\nInference complete. All enhanced images saved to the '{output_dir}' folder.")