import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from seg_dataset import TeethDataModuleSeg
from seg_net import MonaiUNetHRes
from seg_callback import TeethNetImageLoggerSeg
from monai.transforms import Compose
from ManageClass import RandomPickTeethTransform, RandomRotation

from pl_bolts.models.self_supervised import Moco_v2

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

# from azureml.core.run import Run
# run = Run.get_context()


def main(args):


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )
    
    mount_point = args.mount_point

    df_train = pd.read_csv(os.path.join(mount_point, args.csv_train))
    df_val = pd.read_csv(os.path.join(mount_point, args.csv_valid))
    df_test = pd.read_csv(os.path.join(mount_point, args.csv_valid))

    # class_weights = np.load(os.path.join(mount_point, 'train_weights.npy'))
    class_weights = None

    model = MonaiUNetHRes(args, out_channels = 34, class_weights=class_weights, image_size=320, train_sphere_samples=args.train_sphere_samples)

    train_transfrom = Compose([RandomPickTeethTransform(args.property),RandomRotation()])

    teeth_data = TeethDataModuleSeg(df_test=df_test, df_train=df_train, df_val=df_val,mount_point=args.mount_point,
     num_workers = 4,surf_property =args.property,batch_size=args.batch_size,
     train_transform=train_transfrom,val_transform =RandomPickTeethTransform(args.property) , test_transform = RandomPickTeethTransform(args.property))

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)    

    image_logger = TeethNetImageLoggerSeg()

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[early_stop_callback, checkpoint_callback,image_logger],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False, process_group_backend="nccl"),
        num_sanity_val_steps=0,
        profiler=args.profiler
    )
    trainer.fit(model, datamodule=teeth_data, ckpt_path=args.model)

    trainer.test(ckpt_path='best',datamodule=teeth_data)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Teeth challenge Training')
    parser.add_argument('--csv_train', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/Flybycnn/SegmentationTeeth/train_test.csv')    
    parser.add_argument('--csv_valid', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/Flybycnn/SegmentationTeeth/val_test.csv')
    parser.add_argument('--csv_test', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/Flybycnn/SegmentationTeeth/test_test.csv')      
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)    
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--model', help='Model to continue training', type=str, default= None)
    parser.add_argument('--out', help='Output', type=str, default="/home/luciacev/Desktop/Data/Flybycnn/SegmentationTeeth/model_output_train")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/home/luciacev/Desktop/Data/Flybycnn/SegmentationTeeth")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=2)    
    parser.add_argument('--train_sphere_samples', help='Number of training sphere samples or views used during training and validation', type=int, default=4)    
    parser.add_argument('--patience', help='Patience for early stopping', type=int, default=4)
    parser.add_argument('--profiler', help='Use a profiler', type=str, default=None)
    parser.add_argument('--property', help='label of segmentation', type=str, default="PredictedID")
    
    
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default='/home/luciacev/Desktop/Data/Flybycnn/SegmentationTeeth/tensorboard')
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="monai")


    args = parser.parse_args()

    main(args)

