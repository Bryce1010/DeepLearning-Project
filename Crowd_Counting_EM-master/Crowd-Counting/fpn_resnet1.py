import torch.nn as nn
import torch
import math
from torchvision import models
from utils import save_net,load_net
import torch.nn.functional as F
def crop(d, g):
    g_h, g_w = g.size()[2:4]
    d_h, d_w = d.size()[2:4]
    d1 = d[:, :, int(math.floor((d_h - g_h) / 2.0)):int(math.floor((d_h - g_h) / 2.0)) + g_h,
         int(math.floor((d_w - g_w) / 2.0)):int(math.floor((d_w - g_w) / 2.0)) + g_w]
    return d1


## resnet bottleneck
class Bottleneck(nn.Module):
    expansion=4

    def __init__(self,in_planes,planes,stride=1):
        super(Bottleneck,self).__init__()
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=1,bias=False)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)
        self.conv3=nn.Conv2d(planes,self.expansion*planes,kernel_size=1,bias=False)
        self.bn3=nn.BatchNorm2d(self.expansion*planes)

        self.shortcut=nn.Sequential()
        if stride!=1 or in_planes!=self.expansion*planes:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_planes,self.expansion*planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=F.relu(self.bn2(self.conv2(out)))
        out=self.bn3(self.conv3(out))
        out+=self.shortcut(x)
        out=F.relu(out)
        return out








class CSRNet(nn.Module):
    def __init__(self,block,num_blocks,load_weights=False):
        # add two args: block="Bottleneck", num_blocks=[3,4,3,3]
        super(CSRNet, self).__init__()
        self.seen = 0
        self.in_planes=64
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        # Bottom-up layers
        self.layer1=self._make_layer(block,64,num_blocks[0],stride=1)
        self.layer2=self._make_layer(block,128,num_blocks[1],stride=2)
        self.layer3=self._make_layer(block,256,num_blocks[2],stride=2)
        self.layer4=self._make_layer(block,512,num_blocks[3],stride=2)
        
        # Top layer
        self.toplayer=nn.Conv2d(2048,256,kernel_size=1,stride=1,padding=0)

        self.backend_feat  = [512, 512, 512]
        



        #self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        #self.densend = make_layers(self.dense_feat,in_channels = 512, dilation =True)
                
        self.upscore2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore3 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upscore4 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.upscore5 = nn.UpsamplingBilinear2d(scale_factor=8)
        
        self.cd1 = nn.Sequential(nn.Conv2d(64,32,3,padding=1),
            nn.ReLU(inplace=True),
            )
        self.cd2 = nn.Sequential(nn.Conv2d(128,32,3,padding=1),
            nn.ReLU(inplace=True),
            )
        self.cd3 = nn.Sequential(nn.Conv2d(256,32,3,padding=1),
            nn.ReLU(inplace=True),
            )
        self.cd4 = nn.Sequential(nn.Conv2d(512,32,3,padding=1),
            nn.ReLU(inplace=True),
            )
        self.cd5 = nn.Sequential(nn.Conv2d(512,32,3,padding=1),
            nn.ReLU(inplace=True),
            )
        self.fuse = nn.Sequential(nn.Conv2d(112,28,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(28,1,1),
            )
        self.rd5 = nn.Sequential(nn.Conv2d(32, 8, 1),
                                 nn.ReLU(inplace=True))
        self.rd4 = nn.Sequential(nn.Conv2d(40, 8, 1),
                                 nn.ReLU(inplace=True))
        self.rd3 = nn.Sequential(nn.Conv2d(40, 8, 1),
                                 nn.ReLU(inplace=True))
        self.rd2 = nn.Sequential(nn.Conv2d(40, 8, 1),
                                 nn.ReLU(inplace=True))
        self.up5 = nn.ConvTranspose2d(8,8,4,stride=2)
        self.up4 = nn.ConvTranspose2d(8,8,4,stride=2)
        self.up3 = nn.ConvTranspose2d(8,8,4,stride=2)
        self.up2 = nn.ConvTranspose2d(8,8,4,stride=2)
        
        self.dsn1 = nn.Conv2d(40, 1, 1)
        self.dsn2 = nn.Conv2d(40, 1, 1)
        self.dsn3 = nn.Conv2d(40, 1, 1)
        self.dsn4 = nn.Conv2d(40, 1, 1)
        self.dsn5 = nn.Conv2d(32, 1, 1)
        self.dsn6 = nn.Conv2d(4, 1, 1)

        if load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
                
    def forward(self,x,gt,refine_flag):
        pd = (8,8,8,8)
        x= F.pad(x,pd,'constant')
        c1=F.relu(self.bn1(self.conv1(x)))
        c1=F.max_pool2d(c1,kernel_size=3,stride=2,padding=1)
        c2=self.layer1(c1)
        c3=self.layer2(c2)
        c4=self.layer3(c3)
        c5=self.layer4(c4)


        """ conv1 = self.frontend[0:4](x)
        conv2 = self.frontend[4:9](conv1)
        conv3 = self.frontend[9:16](conv2)
        conv4 = self.frontend[16:23](conv3)
        conv5 = self.backend(conv4)
 """
        gt = torch.unsqueeze(gt, 1)

        #p5 = self.cd5(conv5)
        p5=self.toplayer(c5)
        d5 = self.upscore5(self.dsn5(F.relu(p5)))

        d5 = crop(d5, gt)

        p5_up = self.rd5(F.relu(p5))
        #p4_1 = self.cd4(conv4)
        p4_1=self.cd4(c4)
        p4_2 = crop(p5_up, p4_1)
        p4_3 = F.relu(torch.cat((p4_1, p4_2), 1))
        p4 = p4_3
        d4 = self.upscore4(self.dsn4(p4))
        d4 = crop(d4, gt)

        p4_up = self.up4(self.rd4(F.relu(p4)))
        p3_1=self.cd3(c3)
        #p3_1 = self.cd3(conv3)
        p3_2 = crop(p4_up, p3_1)
        p3_3 = F.relu(torch.cat((p3_1, p3_2), 1))
        p3 = p3_3
        d3 = self.upscore3(self.dsn3(p3))
        d3 = crop(d3, gt)

        p3_up = self.up3(self.rd3(F.relu(p3)))
        p2_1=self.cd2(c2)
        #p2_1 = self.cd2(conv2)
        p2_2 = crop(p3_up, p2_1)
        p2_3 = F.relu(torch.cat((p2_1, p2_2), 1))
        p2 = p2_3
        d2 = self.upscore2(self.dsn2(p2))
        d2 = crop(d2, gt)
        #print(gt.shape,d2.shape,d3.shape,d4.shape,d5.shape,gt.shape)
        d6 = self.dsn6(torch.cat((d2, d3, d4, d5), 1))
        
        p_5 = crop(self.upscore5(p5),gt)
        p_4 = crop(self.upscore4(p4),gt)
        p_3 = crop(self.upscore3(p3),gt)
        p_2 = crop(self.upscore2(p2),gt)
        if refine_flag==True:
            feature = torch.cat((p_2,p_3,p_4,p_5),1)
            return d2, d3, d4, d5, d6, feature
#        print(p_2.size(),p3.size(),p4.size(),p5.size(),x.size(),feature.size())

        return d2,d3,d4,d5,d6
    def _make_layer(self,block,planes,num_blocks,stride):
        strides=[stride]+[1]*(num_blocks-1)
        layers=[]
        for stride in strides:
            layers.append(block(self.in_planes,planes,stride))
            self.in_planes=planes*block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)                


if __name__=='__main__':
    model=CSRNet(block=Bottleneck,num_blocks=[2,2,2,2])
    print(model)

