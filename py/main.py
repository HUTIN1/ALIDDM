import argparse
from unittest.mock import patch
from ALIDDM.py.Tools.ALIDDM_utils import *
from ALIDDM.py.DataModule import *
import pandas as pd
from Agent_class import *
from ALIDDM.py.TrainingModule import Model
from pytorch_lightning.loggers import TensorBoardLogger
from monai.transforms import Compose
from ManageClass import RandomPickTeethTransform

# import numpy as np
from scipy import linalg
import pytorch_lightning as pl


def main(args):

    Lower = ['LL7O', 'LL7MB', 'LL7DB', 'LL6O', 'LL6MB', 'LL6DB', 'LL5O', 'LL5MB', 
'LL5DB', 'LL4O', 'LL4MB', 'LL4DB', 'LL3O', 'LL3MB', 'LL3DB', 'LL2O', 'LL2MB',
 'LL2DB', 'LL1O', 'LL1MB', 'LL1DB', 'LR1O', 'LR1MB', 'LR1DB', 'LR2O', 'LR2MB', 
 'LR2DB', 'LR3O', 'LR3MD', 'LR3DB', 'LR4O', 'LR4MB', 'LR4DB', 'LR5O', 'LR5MB',
  'LR5DB', 'LR6O', 'LR6MB', 'LR6DB', 'LR7O', 'LR7MB', 'LR7DB', 'LL7R', 'LL6R', 
  'LL5R', 'LL4R', 'LL3R', 'LL2R', 'LL1R']

    Upper = ['UL7O', 'UL7MB', 'UL7DB', 'UL6O', 'UL6MB', 'UL6DB', 'UL5O', 'UL5MB', 
'UL5DB', 'UL4O', 'UL4MB', 'UL4DB', 'UL3O', 'UL3MB', 'UL3DB', 'UL2O', 'UL2MB', 
'UL2DB', 'UL1O', 'UL1MB', 'UL1DB', 'UR1O', 'UR1MB', 'UR1DB', 'UR2O', 'UR2MB', 
'UR2DB', 'UR3O', 'UR3MD', 'UR3DB', 'UR4O', 'UR4MB', 'UR4DB', 'UR5O', 'UR5MB', 
'UR5DB', 'UR6O', 'UR6MB', 'UR6DB', 'UR7O', 'UR7MB', 'UR7DB', 'UL7R', 'UL6R', 
'UL5R', 'UL4R', 'UL3R', 'UL2R', 'UL1R']


    sphere_points_L = ([0,0,1],
                    np.array([0.5,0.,1.0])/linalg.norm([0.5,0.5,1.0]),
                    np.array([-0.5,0.,1.0])/linalg.norm([-0.5,-0.5,1.0]),
                    np.array([0,0.5,1])/linalg.norm([1,0,1]),
                    np.array([0,-0.5,1])/linalg.norm([0,1,1])
                    )
    sphere_points_U = ([0,0,-1],
                    np.array([0.5,0.,-1])/linalg.norm([0.5,0.5,-1]),
                    np.array([-0.5,0.,-1])/linalg.norm([-0.5,-0.5,-1]),
                    np.array([0,0.5,-1])/linalg.norm([1,0,-1]),
                    np.array([0,-0.5,-1])/linalg.norm([0,1,-1])
                    )

    # GV.DEVICE = torch.device(f"cuda:{args.num_device}" if torch.cuda.is_available() else "cpu")
    
    if torch.has_cuda :
        device = torch.device(f"cuda:{args.num_device}")
    else :
        device = torch.device("cpu")


    unique_ids = []
    if args.jaw == 'L':
        for teeth in args.lst_label_l:
            unique_ids.append(int(teeth))
    else :
        for teeth in args.lst_label_u:
            unique_ids.append(int(teeth))



    
    if args.jaw == "L":
        camera_position= np.array(sphere_points_L)
        lst_label = args.lst_label_l
        lst_landmarks = Lower
    else:
        camera_position = np.array(sphere_points_U)
        lst_label = args.lst_label_u
        lst_landmarks = Upper
  
    phong_renderer,mask_renderer = GenPhongRenderer(args.image_size,args.blur_radius,args.faces_per_pixel, device=device)



    agent = Agent(
        renderer=phong_renderer,
        renderer2=mask_renderer,
        radius=args.sphere_radius,
        device = device,
        camera_position=camera_position
    )


    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)   
    
    if not os.path.exists(args.dir_models):
            os.makedirs(args.dir_models)


    transforms = Compose([RandomRotation(), RandomPickTeethTransform(surf_property=args.surf_property, unique_ids=unique_ids)])

    teeth_data = TeethDataModule(df = args.csv_file,
                                    dataset_dr = args.dir_patients,
                                    batch_size = args.batch_size,
                                    train_transform = transforms,
                                    device = device,
                                    jaw = args.jaw,
                                    lst
            )
    net = Model(lst_label,agent,args)
    trainer = pl.Trainer(
        logger = logger ,
        max_epochs = args.max_epoch,
        accelerator = 'gpu',
        devices = 1 , #torch.cuda.device_count(),
        callbacks = PrintCallback()

    )

    trainer.fit(net,datamodule=teeth_data)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automatic Landmark Identification on Digital Dental Model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dir_patients', type=str, help='Input directory with the meshes',default='/patients')
    parser.add_argument('--csv_file', type=str, help='CSV of the data split',default='/data_split/Lower/data_splitfold3.csv')


    #Environment
    parser.add_argument('-j','--jaw',type=str,help="Prepare the data for uper or lower landmark training (ex: L U)", default="L")
    parser.add_argument('-sr', '--sphere_radius', type=float, help='Radius of the sphere with all the cameras', default=0.2)
    parser.add_argument('--lst_label_l', type=list, help='label of the teeth',default=(["18","19","20","21","22","23","24","25","26","27","28","29","30","31"]))
    parser.add_argument('--lst_label_u', type=list, help='label of the teeth',default=(["2","3","4","5","6","7","8","9","10","11","12","13","14","15"]))


    #Training data
    parser.add_argument('--num_device',type=int, help='cuda:0 or cuda:1', default=1)
    parser.add_argument('--image_size',type=int, help='size of the picture', default=224)
    parser.add_argument('--blur_radius',type=int, help='blur raius', default=0)
    parser.add_argument('--faces_per_pixel',type=int, help='faces per pixels', default=1)
    
    parser.add_argument('--batch_size', type=int, help='Batch size', default=3)
    parser.add_argument('-nc', '--num_classes', type=int, help='number of classes', default=4)
    parser.add_argument('--surf_property', type=str, help='the name of segmentation', default='PredictedID')


    #Training param
    parser.add_argument('--max_epoch', type=int, help='Number of training epocs', default=300)
    parser.add_argument('--val_freq', type=int, help='Validation frequency', default=1)
    parser.add_argument('--val_percentage', type=int, help='Percentage of data to keep for validation', default=10)
    parser.add_argument('--test_percentage', type=int, help='Percentage of data to keep for test', default=20)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=1e-4)
    # input_param.add_argument('-nw', '--nbr_worker', type=int, help='Number of worker', default=0)

    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="monai")

   

    parser.add_argument('--dir_models', type=str, help='Output directory with all the networks',default='/models/Lower/test_unique_models_csv3')

    
    args = parser.parse_args()
    main(args)