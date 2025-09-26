## CS7180 - Advanced Perception - Final Training Script
## DATE: 09/21/2025
## Gautham Ramkumar & Sai Vamsi Rithvik Allanka

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

from model import FinalModel
from losses import ssim_loss, l1_loss
from dataloader import get_train_dataset, get_val_dataset, IMAGE_SIZE

## Custom PSNR Metric for evaluation 
class PSNRMetric(keras.metrics.Metric):
    def __init__(self, name="psnr", **kwargs):
        super().__init__(name=name, **kwargs)
        self.psnr = keras.metrics.Mean(name="psnr_mean")
    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_psnr = tf.image.psnr(y_true, y_pred, max_val=1.0)
        self.psnr.update_state(batch_psnr, sample_weight=sample_weight)
    def result(self):
        return self.psnr.result()
    def reset_state(self):
        self.psnr.reset_state()

## Custom class for initializing the model, losses, and metrics
class SupervisedTrainer(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = FinalModel()

        ## Loss functions
        self.l1_loss = keras.losses.MeanAbsoluteError()
        self.ssim_loss = ssim_loss

    ## Compile function to set optimizer and initialize metrics
    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        # Initialize metric trackers
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.gray_loss_tracker = keras.metrics.Mean(name="gray_loss")
        self.enhance_loss_tracker = keras.metrics.Mean(name="enhance_loss")
        self.psnr_tracker = PSNRMetric(name="psnr")
        self.mse_tracker = keras.metrics.MeanSquaredError(name="mse")

    ## Function to call the model for inference
    def call(self, inputs):
        return self.model(inputs)

    ## Function to return the metrics being tracked
    @property
    def metrics(self):
        return [
        self.total_loss_tracker, self.gray_loss_tracker, self.enhance_loss_tracker,
        self.psnr_tracker, self.mse_tracker
        ]
        
    ## Custom Training Step - calculating losses and applying custom weights for each loss component
    def train_step(self, data):
        low_light_image, high_light_image = data
        high_light_grayscale = tf.image.rgb_to_grayscale(high_light_image)
        with tf.GradientTape() as tape:
            ## Get all three outputs from the model
            predicted_grayscale, _, final_enhanced_image = self.model(low_light_image, training=True)
            
            ## Compute losses for both heads
            ## Loss for the grayscale head (L1 + SSIM)
            loss_gray = l1_loss(high_light_grayscale, predicted_grayscale) + ssim_loss(high_light_grayscale, predicted_grayscale)
            
            ## Loss for the final enhancement (L1 + SSIM)
            loss_enhance = l1_loss(high_light_image, final_enhanced_image) + ssim_loss(high_light_image, final_enhanced_image)
            
            ## Total loss is a weighted sum of both losses
            total_loss = loss_gray + loss_enhance

        ## Apply gradients to update model weights
        gradients = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        
        ## Update trackers for losses and metrics
        self.total_loss_tracker.update_state(total_loss)
        self.gray_loss_tracker.update_state(loss_gray)
        self.enhance_loss_tracker.update_state(loss_enhance)
        return {m.name: m.result() for m in self.metrics if "psnr" not in m.name and "mse" not in m.name}
    
    ## Custom Testing Step - for validation and testing phases with metrics tracking and custom loss calculations
    def test_step(self, data):
        low_light_image, high_light_image = data
        high_light_grayscale = tf.image.rgb_to_grayscale(high_light_image)
        
        predicted_grayscale, _, final_enhanced_image = self.model(low_light_image, training=False)
        
        ## Compute losses for both heads same as training step
        loss_gray = self.l1_loss(high_light_grayscale, predicted_grayscale) + ssim_loss(high_light_grayscale, predicted_grayscale)
        loss_enhance = self.l1_loss(high_light_image, final_enhanced_image) + ssim_loss(high_light_image, final_enhanced_image)
        total_loss = loss_gray + loss_enhance

        ## Update all trackers for validation
        self.total_loss_tracker.update_state(total_loss)
        self.gray_loss_tracker.update_state(loss_gray)
        self.enhance_loss_tracker.update_state(loss_enhance)
        self.psnr_tracker.update_state(high_light_image, final_enhanced_image)
        self.mse_tracker.update_state(high_light_image, final_enhanced_image)
        return {m.name: m.result() for m in self.metrics}

## Main execution block to load data, initialize trainer, and start training
if __name__ == "__main__":
    ## Load the training and validation datasets
    train_dataset = get_train_dataset()
    val_dataset = get_val_dataset()

    ## Initialize the model, compile it with Adam optimizer, and set up checkpointing
    trainer = SupervisedTrainer()
    trainer.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))
    trainer.build(input_shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="final_supervised_model.weights.h5", 
        save_weights_only=True,
        monitor="val_psnr",
        mode="max",
        save_best_only=True
    )

    history = trainer.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=[model_checkpoint_callback]
    )

    ## Find the epoch with the best validation PSNR
    best_psnr_epoch = np.argmax(history.history['val_psnr']) + 1
    best_psnr = history.history['val_psnr'][best_psnr_epoch - 1]
    best_loss_at_psnr_epoch = history.history['val_total_loss'][best_psnr_epoch - 1]

    ## Print the best model performance summary
    print("\n" + "="*50)
    print("             BEST MODEL PERFORMANCE")
    print("="*50)
    print(f"-> Best Validation PSNR: {best_psnr:.2f} dB")
    print(f"   (Achieved at epoch {best_psnr_epoch})")
    print(f"-> Validation Loss at this epoch: {best_loss_at_psnr_epoch:.4f}")
    print("="*50 + "\n")

    ## Plotting Training and Validation Metrics
    def plot_result(item):
        plt.plot(history.history[item], label=item)
        if "val_" + item in history.history:
            plt.plot(history.history["val_" + item], label="val_" + item)
            plt.xlabel("Epochs")
        plt.ylabel(item)
        plt.title(f"Train and Validation {item} Over Epochs", fontsize=14)
        plt.legend()
        plt.grid()
        plt.show()

    plot_result("total_loss")
    plot_result("val_psnr")
    plot_result("val_mse")