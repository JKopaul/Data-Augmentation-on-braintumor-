import torch
import torch.nn as nn

def double_conv(in_c,out_c):
  conv=nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=3,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c,out_c,kernel_size=3,stride=1,padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
            )
  return conv

def crop_img(tensor,target_tensor):
  target_size=target_tensor.size()[2]
  tensor_size=tensor.size()[2]
  delta=tensor_size-target_size
  delta=delta //2
  return tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta]


class UNet2d(nn.Module):
  def __init__(self):
    super(UNet2d,self).__init__()

    self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
    self.downconv1=double_conv(1,64)
    self.downconv2=double_conv(64,128)
    self.downconv3=double_conv(128,256)
    self.downconv4=double_conv(256,512)
    self.downconv5=double_conv(512,1024)


    self.uptranspose1=nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=2,stride=2)
    self.upconv1=double_conv(1024,512)

    self.uptranspose2=nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=2,stride=2)
    self.upconv2=double_conv(512,256)

    self.uptranspose3=nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=2)
    self.upconv3=double_conv(256,128)

    self.uptranspose4=nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2)
    self.upconv4=double_conv(128,64)

    self.out=nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1)


  def forward(self,images):
    #Encoder
    x1=self.downconv1(images)
    x2=self.maxpool(x1)
    x3=self.downconv2(x2)
    x4=self.maxpool(x3)
    x5=self.downconv3(x4)
    x6=self.maxpool(x5)
    x7=self.downconv4(x6)
    x8=self.maxpool(x7)
    x9=self.downconv5(x8)

    #Decoder

    x=self.uptranspose1(x9)
    y=crop_img(x7,x)
    x=self.upconv1(torch.cat([y,x],1))

    x=self.uptranspose2(x)
    y=crop_img(x5,x)
    x=self.upconv2(torch.cat([y,x],1))

    x=self.uptranspose3(x)
    y=crop_img(x3,x)
    x=self.upconv3(torch.cat([y,x],1))

    x=self.uptranspose4(x)
    y=crop_img(x1,x)
    x=self.upconv4(torch.cat([y,x],1))

    x=self.out(x)
    #print("the output shape:",x.shape)
    return x