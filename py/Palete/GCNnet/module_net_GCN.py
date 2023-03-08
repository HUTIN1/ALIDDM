import pytorch_lightning as pl
from net_GCN import MeshSeg, MeshSegbis
import torch
from net_GCN import GraphFeatureEncoder, get_mlp_layers
from torch import nn
from time import sleep




class GCNNet(pl.LightningModule):
    def __init__(self, lr=1e-4,batch_size=1,num_classes=2,in_features = 3) -> None:
        super(GCNNet,self).__init__()

        self.save_hyperparameters()
        # self.register_buffer("num_classes", torch.tensor(num_classes))
        # self.register_buffer('in_features',torch.tensor(in_features))

        self.input_encoder = None


        # model_params = dict(
        #     in_features=3,
        #     encoder_features=16,
        #     conv_channels=[32, 64, 128, 64],
        #     encoder_channels=[16],
        #     decoder_channels=[32],
        #     num_classes=2,
        #     num_heads=2,
        #     apply_batch_norm=True,
        # )
        self.lr = lr 
        self.net = MeshSeg( in_features=in_features,
            encoder_features=16,
            conv_channels=[32, 64, 128, 64],
            encoder_channels=[16],
            decoder_channels=[32],
            num_classes=num_classes,
            num_heads=num_classes,
            apply_batch_norm=True)




        self.loss = torch.nn.CrossEntropyLoss()
        self.batch_size = batch_size

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr = self.lr)
        return optimizer
    
    def to(self,device = None):
        return super().to(device)


    def forward(self,data):
        return self.net(data)


    def training_step(self, train_batch , batch_idx) :
        train_batch.to(self.device)
        out = self(train_batch)
        loss = self.loss(out,train_batch.segmentation_labels.squeeze())

        self.log('train_loss',loss, batch_size=self.batch_size)
        acc = self.accuracy(out,train_batch.segmentation_labels)
        self.log('train_acc',acc,batch_size=self.batch_size)

        return loss
    

    def validation_step(self, val_batch , batch_idx) :
        val_batch.to(self.device)
        out = self(val_batch)
        loss = self.loss(out,val_batch.segmentation_labels.squeeze())

        self.log('val_loss',loss, batch_size=self.batch_size)
        acc = self.accuracy(out,val_batch.segmentation_labels)
        self.log('val_acc',acc,batch_size=self.batch_size)


    def test_step(self, test_batch, batch_idx):
        test_batch.to(self.device)
        out = self(test_batch)
        loss = self.loss(out,test_batch.segmentation_labels.squeeze())
        acc = self.accuracy(out,test_batch.segmentation_labels)

        return {'test_loss':loss, 'test_accuracy':acc} 
    

    def accuracy(self,predictions, gt_seg_labels):
        predicted_seg_labels = predictions.argmax(dim=-1, keepdim=True)
        if predicted_seg_labels.shape != gt_seg_labels.shape:
            raise ValueError("Expected Shapes to be equivalent")
        correct_assignments = (predicted_seg_labels == gt_seg_labels).sum()
        num_assignemnts = predicted_seg_labels.shape[0]
        return float(correct_assignments / num_assignemnts)
    


        
