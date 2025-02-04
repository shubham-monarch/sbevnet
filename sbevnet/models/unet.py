from __future__ import print_function
import torch.nn as nn
import torch 
import torch.nn.functional as F
from helpers import get_logger


class contracting(nn.Module):
    def __init__(self , n_channels=3 ):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(n_channels , 64, 3, stride=1, padding=1) , nn.BatchNorm2d(64) , nn.ReLU(), nn.Conv2d(64, 64, 3, stride=1, padding=1) , nn.BatchNorm2d(64), nn.ReLU())

        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1) , nn.BatchNorm2d(128) , nn.ReLU(), nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128) , nn.ReLU())

        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1) , nn.BatchNorm2d(256 ) , nn.ReLU(), nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())

        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1),  nn.BatchNorm2d(512) , nn.ReLU(), nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU())

        self.layer5 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=1, padding=1) , nn.BatchNorm2d(512) , nn.ReLU(), nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.BatchNorm2d(512) , nn.ReLU())

        self.down_sample = nn.MaxPool2d(2, stride=2)


    def forward(self, X):
        X1 = self.layer1(X)
        X2 = self.layer2(self.down_sample(X1))
        X3 = self.layer3(self.down_sample(X2))
        X4 = self.layer4(self.down_sample(X3))
        X5 = self.layer5(self.down_sample(X4))
        return X5, X4, X3, X2, X1



class expansive(nn.Module):
    def __init__(self , inp_shape=128 , n_channels=3 ):
        super().__init__()
        
        self.inp_shape = inp_shape  # width of the image 

        self.logger = get_logger("expansive")

        self.layer1 = nn.Conv2d(128, n_channels  , 3, stride=1, padding=1)

        self.layer2 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1),
                                    nn.ReLU(), nn.Conv2d(128, 128, 3, stride=1, padding=1),
                                    nn.ReLU())

        self.layer3 = nn.Sequential(nn.Conv2d(256, 128, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU())

        self.layer4 = nn.Sequential(nn.Conv2d(512, 256, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU())

        self.layer5 = nn.Sequential(nn.Conv2d(1024, 512, 3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.ReLU())

        self.up_sample_54 = nn.ConvTranspose2d(512, 512, 2, stride=2)

        self.up_sample_43 = nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0)

        self.up_sample_32 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.up_sample_21 = nn.ConvTranspose2d(128, 64, 2, stride=2)


    def forward(self, X5, X4, X3, X2, X1):
        
        # self.logger.info(f'=================')
        # self.logger.info(f'CKPT-1')
        # self.logger.info(f'=================\n')

        X = self.up_sample_54(X5)

        # self.logger.info(f'=================')
        # self.logger.info(f'CKPT-2')
        # self.logger.info(f'=================\n')

        # self.logger.info(f'=================')
        # self.logger.info(f'X.shape: {X.shape}')
        # self.logger.info(f'X4.shape: {X4.shape}')
        # self.logger.info(f'=================\n')

        X4 = torch.cat([X, X4], dim=1)

        # self.logger.info(f'=================')
        # self.logger.info(f'CKPT-3')
        # self.logger.info(f'=================\n')

        X4 = self.layer5(X4)

        # self.logger.info(f'=================')
        # self.logger.info(f'CKPT-4')
        # self.logger.info(f'=================\n')

        X = self.up_sample_43(X4)

        # self.logger.info(f'=================')
        # self.logger.info(f'CKPT-5')
        # self.logger.info(f'=================\n')
        
        if self.inp_shape == 100:
            X = F.pad(X, (0,1,0,1), mode='replicate')

        # self.logger.info(f'=================')
        # self.logger.info(f'CKPT-6')
        # self.logger.info(f'=================\n')

        X3 = torch.cat([X, X3], dim=1)
        
        # self.logger.info(f'=================')
        # self.logger.info(f'CKPT-7')
        # self.logger.info(f'=================\n')

        X3 = self.layer4(X3)

        # self.logger.info(f'=================')
        # self.logger.info(f'CKPT-8')
        # self.logger.info(f'=================\n')

        X = self.up_sample_32(X3)
        
        # self.logger.info(f'=================')
        # self.logger.info(f'CKPT-9')
        # self.logger.info(f'=================\n')

        X2 = torch.cat([X, X2], dim=1)

        # self.logger.info(f'=================')
        # self.logger.info(f'CKPT-10')
        # self.logger.info(f'=================\n')

        X2 = self.layer3(X2)

        # self.logger.info(f'=================')
        # self.logger.info(f'CKPT-11')
        # self.logger.info(f'=================\n')

        X = self.up_sample_21(X2)

        # self.logger.info(f'=================')
        # self.logger.info(f'CKPT-12')
        # self.logger.info(f'=================\n')

        X1 = torch.cat([X, X1], dim=1)

        # self.logger.info(f'=================')
        # self.logger.info(f'CKPT-13')
        # self.logger.info(f'=================\n')

        X1 = self.layer2(X1)

        # self.logger.info(f'=================')
        # self.logger.info(f'CKPT-14')
        # self.logger.info(f'=================\n')

        X = self.layer1(X1)

        # self.logger.info(f'=================')
        # self.logger.info(f'CKPT-15')
        # self.logger.info(f'=================\n')

        return X


class UNet(nn.Module):
    def __init__(self , inp_shape=128, n_channels=3 ):
        super().__init__()
        self.down = contracting( n_channels=n_channels)
        self.up = expansive(inp_shape=inp_shape , n_channels=n_channels)
        self.logger = get_logger("UNet")
    def forward(self, X):

        # self.logger.error(f'=================')
        # self.logger.error(f'CKPT-1')
        # self.logger.error(f'=================\n')

        X5, X4, X3, X2, X1 = self.down(X)


        # self.logger.error(f'=================')
        # self.logger.error(f'CKPT-2')
        # self.logger.error(f'=================\n')

        # self.logger.info(f'=================')
        # self.logger.info(f'X5.shape: {X5.shape}')
        # self.logger.info(f'X4.shape: {X4.shape}')
        # self.logger.info(f'X3.shape: {X3.shape}')
        # self.logger.info(f'X2.shape: {X2.shape}')
        # self.logger.info(f'X1.shape: {X1.shape}')
        # self.logger.info(f'=================\n')

        X = self.up(X5, X4, X3, X2, X1)

        # self.logger.error(f'=================')
        # self.logger.error(f'CKPT-3')
        # self.logger.error(f'=================\n')

        return X

