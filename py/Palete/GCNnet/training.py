import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from module_net_GCN import GCNNet
from dataset import DataModuleGCN
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
# from landmark_callback import TeethNetImageLoggerLm
from pytorch_lightning import Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
import torch
import os
from torch_geometric.transforms import FaceToEdge
from ManageClass import UnitSurfTransform
from typing import Union
def main(args):

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename=f'{args.landmark}_radius={args.radius}'+'_{epoch}-{val_loss:.3f}',
        save_top_k=2,
        monitor='val_loss'
    )

    if args.load_checkpoint :
        model = GCNNet.load_from_checkpoint(args.load_checkpoint)
    else :
        model = GCNNet(lr = args.lr, batch_size=args.batch_size,num_classes = 2,in_features=6)

    transform = FaceToEdge(remove_faces=False)

    teeth_data = DataModuleGCN(train_csv=args.csv_train,
                                val_csv= args.csv_valid,
                                test_csv=args.csv_test,
                                landmark=args.landmark[0],
                                num_worker=args.num_workers,
                                batch_size=args.batch_size,
                                transform = transform,
                                radius=args.radius,
                                surf_transform=UnitSurfTransform())
    
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")


    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)    


    # image_logger = TeethNetImageLoggerLm(log_steps=args.log_every_n_steps)



    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[early_stop_callback, checkpoint_callback],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False, process_group_backend="nccl"),
        num_sanity_val_steps=0,
        profiler=args.profiler   
    )

    trainer.fit(model, datamodule=teeth_data, ckpt_path=args.model)
    trainer.test(ckpt_path='best', datamodule=teeth_data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GCN neural network Identification Landmark')
    parser.add_argument('--csv_train', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/data/csv/train_LL1O.csv')    
    parser.add_argument('--csv_valid', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/data/csv/val_LL1O.csv')
    parser.add_argument('--csv_test', help='CSV with column surf', type=str, default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/data/csv/test_LL1O.csv')  
    parser.add_argument('--mouth_path',default='/home/luciacev/Desktop/Data/IOSReg/renamed_segmented/')    
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)    
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=500)    
    parser.add_argument('--model', help='Model to continue training', type=str, default= None)
    parser.add_argument('--out', help='Output', type=str, default="/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/GCN/model/during_train")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=30)       
    parser.add_argument('--patience', help='Patience for early stopping', type=int, default=50)
    parser.add_argument('--profiler', help='Use a profiler', type=str, default=None)
    parser.add_argument('--landmark',help='name of landmark to found',type= Union [str , list] , default=["LL1O"])
    parser.add_argument('--radius',help='radius of landmark on mesh',default=0.1)
    parser.add_argument('--load_checkpoint')
    
    
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default='/home/luciacev/Desktop/Data/ALI_IOS/landmark/Training/GCN/tensorboard')
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="monai")

    args = parser.parse_args()

    main(args)