#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf


from PIL import Image

import os
import glob
import numpy as np

from sklearn import model_selection
import segmentation_models as sm

from DataLoader import DataLoader, make_augmentation
from Model import unet_model
from util import dict_to_config

import argparse


class SegmentationTraining(object):
    def __init__(config):
        self._config = config

    def _get_datasets():
        base_dataset_dir = self._config.DATASET_PATH
        ims_dir = os.path.join(base_dataset_dir, "imgs")
        labels_dir = os.path.join(base_dataset_dir, "labels")
        img_files = sorted(glob.glob(ims_dir + "/*.*"))
        mask_files = sorted(glob.glob(labels_dir + "/*.png"))
        print (f"Masks count : {len(mask_files)};\nImgs count : {len(img_files)};")

        dataset = list(zip(img_files, mask_files))
        train_dataset, test_dataset = model_selection.train_test_split(dataset, test_size=0.2, random_state=0)
        print(f"Train dataset size {len(train_dataset)}; Test dataset size {len(test_dataset)}")
        return train_dataset, test_dataset


    def _get_data_loader(train_dataset, test_dataset, batch_size=24, output_size=(256, 256)):
        training_augmentation = make_augmentation(output_size=output_size, is_validation=False)
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, augmentation_fn=training_augmentation, output_size=output_size, shuffle=True)

        validation_augmentation = make_augmentation(output_size=output_size, is_validation=True)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, augmentation_fn=validation_augmentation, output_size=output_size, shuffle=False)
        return train_data_loader, test_data_loader


    def _get_model():
        OUTPUT_CHANNELS = 1
        model = unet_model(OUTPUT_CHANNELS)

        if weights_path in self._config:
            model.load_weights(self._config.weights_path)
        
        optim = tf.keras.optimizers.Adam(self._config.LR)
        
        dice_loss = sm.losses.DiceLoss()
        # focal_loss = sm.losses.BinaryFocalLoss()
        # total_loss = dice_loss + (1 * focal_loss)
        total_loss = dice_loss

        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

        # compile keras model with defined optimozer, loss and metrics
        model.compile(optim, total_loss, metrics)
        return model

    def _launch_training(model, train_data_loader, test_data_loader):        
        project_launch_time = time.strftime("%Y_%m_%d_%H_%M",time.localtime())

        checkpoint_file_format = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint_path = os.path.join("experiments", project_launch_time, config.exp_name, "checkpoint", checkpoint_file_format)
        logs_dir = os.path.join("experiments", project_launch_time, config.exp_name, "logs/")
        
        print(f"Checkpoints path:{checkpoint_path}\nLogs path:{logs_dir}")

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(checkpoint_path , save_weights_only=True, save_best_only=True, mode='min'),
            tf.keras.callbacks.ReduceLROnPlateau(),
            tf.keras.callbacks.TensorBoard(log_dir=logs_dir)
        ]

        history = model.fit(
            x = train_data_loader, 
            steps_per_epoch=len(train_data_loader), 
            epochs=config.EPOCHS, 
            callbacks=callbacks, 
            validation_data=test_data_loader, 
            validation_steps=len(test_data_loader),
            use_multiprocessing=False,
            workers=1
        ) 

    def train():
        train_dataset, test_dataset = self._get_datasets()
        train_dataloader, test_dataloader = self._get_data_loader(train_dataset, test_dataset, batch_size=BATCH_SIZE)
        model = self._get_model()
        self._launch_training(model, train_dataloader, test_dataloader)


def main():
    dict_config = {
        "EXP_NAME" : "training",

        "LR" : 0.0001,
        "EPOCHS" : 40,
        "BATCH_SIZE" : 24,        

        "DATASET_PATH" : "/storage/supervisely/processed",
        "OUTPUT_DIR" : '/storage/training_result/'
    }

    config = dict_to_config(config)
    training = SegmentationTraining(config)
    training.train()


if __name__ == "__main__":
    main()