"""
Simple GAN using fully connected layers
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
import nibabel as nib
import glob
from scipy import ndimage
from torch.utils.data import Dataset


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


class Segmentation_Dataset(Dataset):
    def __init__(self, imgflder):
        self.imgflder = imgflder

    def __len__(self):
        return len(self.imgflder)

    def __getitem__(self, idx):
        # TO IMAGE
        img_path = self.imgflder[idx]
        a = nib.load(img_path)
        img = a.get_fdata()

        # Name of Image
        name_strr = self.imgflder[idx]
        name_str = name_strr.rpartition("/")
        name_img = name_str[2]

        # to normalize
        mn = img.min()
        mx = img.max()
        img = (img - mn) / (mx - mn)

        # Resize volume to 560 x 640 x 200

        desired_depth = 155
        desired_width = 240
        desired_height = 240
        # Get current depth
        current_depth = img.shape[2]
        current_width = img.shape[1]
        current_height = img.shape[0]
        # Compute depth factor
        depth = current_depth / desired_depth
        width = current_width / desired_width
        height = current_height / desired_height
        depth_factor = 1 / depth
        width_factor = 1 / width
        height_factor = 1 / height

        # Resize across z-axis
        img = ndimage.zoom(img, (height_factor, width_factor, depth_factor), order=1)

        img = np.transpose(img, (2, 0, 1)).astype(np.float64)

        img = torch.Tensor(img)
        # To make 360 x360 x 200
        # img=img[:,100:460,140:500]

        return img, name_img


train_Imagefolder_path = "/Users/jins_jr/Main Project/TrainImage/*"
train_Imagefolder = sorted(glob.glob(train_Imagefolder_path))
len_Image_folder_train = len(train_Imagefolder)
print("No. of Images Train_Imagefolder:", len_Image_folder_train)

trainset = Segmentation_Dataset(train_Imagefolder, )
print(f'length of trainset:{len(trainset)}')

# Hyperparameters etc.
device = "cpu"
lr = 3e-4
z_dim = 64
image_dim = 155 * 240 * 240
batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

loader=DataLoader(trainset,batch_size=1,shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
# writer_fake = SummaryWriter(f"logs/fake")
# writer_real = SummaryWriter(f"logs/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 240*240*155).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            # with torch.no_grad():
            #     fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
            #     data = real.reshape(-1, 1, 28, 28)
            #     img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
            #     img_grid_real = torchvision.utils.make_grid(data, normalize=True)
            #     step += 1
