##
# v0: 5 layers
# v1: only use the bridge layer for classification.
##

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from SDFY_project.unet_model.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,
                 height, width,
                 n_seg_classes=1,
                 device=torch.device('cuda')):
        super(UNet, self).__init__()

        self.device = device
        self.n_classes = n_classes

        # With this network depth, there is a minimum image size
        if height < 256 or width < 256:
            raise ValueError('Minimum input image size is 256x256, got {}x{}'.\
                             format(height, width))

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.down5 = down(512, 512)
        self.down6 = down(512, 512)
        self.down7 = down(512, 512)
        self.down8 = down(512, 512, normaliz=False)

        self.up1 = up(1024, 512)
        self.up2 = up(1024, 512)
        self.up3 = up(1024, 512)
        self.up4 = up(1024, 512)
        self.up5 = up(1024, 256)
        self.up6 = up(512, 128)
        self.up7 = up(256, 64)
        self.up8 = up(128, 64, activ=False)
        self.outc = outconv(64, n_seg_classes)

        self.out_nonlin = nn.Sigmoid()   # to solve the predict map wired problem @ 20190924 Xing
        # self.softmax = nn.LogSoftmax(dim=1)

        steps = 5
        height_mid_features = height // (2 ** steps)
        width_mid_features = width // (2 ** steps)
        self.branch_1 = nn.Sequential(nn.Linear(height_mid_features * \
                                                width_mid_features * \
                                                512,
                                                64),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(p=0.5))
        self.branch_2 = nn.Sequential(nn.Linear(height * width, 64),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(p=0.5))

        if n_classes is None:

            self.regressor = nn.Sequential(nn.Linear(64 + 64, 1),
                                           nn.ReLU())
        else:
            # self.fc1 = nn.Sequential(nn.Linear(64+64,16,bias=False),nn.ReLU(inplace=True))
            self.fc = nn.Linear(64+64,n_classes,bias=False)

        self.softmax = nn.LogSoftmax(dim=1)


        # This layer is not connected anywhere
        # It is only here for backward compatibility
        self.lin = nn.Linear(1, 1, bias=False)

    def forward(self, x):

        batch_size = x.shape[0]
        # print(batch_size)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # x6 = self.down5(x5)
        # x7 = self.down6(x6)
        # x8 = self.down7(x7)
        x8 = x5
        x9 = self.down8(x8)


        x = self.up1(x9, x8)
        # x = self.up2(x, x7)
        # x = self.up3(x, x6)
        # x = self.up4(x, x5)
        x = self.up5(x, x4)
        x = self.up6(x, x3)
        x = self.up7(x, x2)
        x = self.up8(x, x1)

        x= self.outc(x)
        x = self.out_nonlin(x)      # do not use the sigmoid @20190924 by Xing

        # Reshape Bx1xHxW -> BxHxW
        # because probability map is real-valued by definition
        x = x.squeeze(1)

        x9_flat = x9.view(batch_size, -1)
        x_flat = x.view(batch_size, -1)

        # print(x9_flat.shape)

        x10_flat = self.branch_1(x9_flat)
        x_flat = self.branch_2(x_flat)

        # final_features = torch.cat((x_flat, x10_flat), dim=1)
        final_features = torch.cat((x10_flat, x10_flat), dim=1)
        if self.n_classes is None:

            regression = self.regressor(final_features)

            return x, regression

        else:

            if self.n_classes == 1:
                classification = self.out_nonlin(self.fc(final_features))
            else:
                classification = self.softmax(self.fc(final_features))

            return x,classification


if __name__ == "__main__":

    num_classes = 5
    input_tensor = torch.autograd.Variable(torch.rand(6, 3, 512, 512)).cuda()
    # model = resnet50(class_num=num_classes)
    model = UNet(n_channels=3,n_classes=num_classes,height=512,width=512).cuda()
    output = model(input_tensor)

    print(output)

