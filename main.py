import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


image_size = 64
batch_size = 64
latent_dim = 100
epochs = 3
sample_dir = 'samples'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(sample_dir, exist_ok=True)



data_root = "/Users/hemanthraman/.cache/kagglehub/datasets/splcher/animefacedataset/versions/3"
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

raw_dataset = datasets.ImageFolder(root=data_root, transform=transform)


class SafeDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        img, label = self.dataset[index]
        if img.shape[-1] < 64 or img.shape[-2] < 64:
            return self.__getitem__((index + 1) % len(self.dataset))
        return img, label

    def __len__(self):
        return len(self.dataset)

dataset = SafeDataset(raw_dataset)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, img):
        return self.model(img).view(-1, 1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG = Generator().to(device)
netD = Discriminator().to(device)
netG.apply(weights_init)
netD.apply(weights_init)


criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

print("Starting Training...")
for epoch in range(epochs):
    loop = tqdm(dataloader, desc=f"[Epoch {epoch + 1}/{epochs}]")
    for i, (real_imgs, _) in enumerate(loop):
        real_imgs = real_imgs.to(device)
        if real_imgs.shape[-1] < 64 or real_imgs.shape[-2] < 64:
            continue
        bs = real_imgs.size(0)

        real_labels = torch.ones(bs, 1, device=device)
        fake_labels = torch.zeros(bs, 1, device=device)

        optimizerD.zero_grad()
        outputs_real = netD(real_imgs)
        d_loss_real = criterion(outputs_real, real_labels)

        noise = torch.randn(bs, latent_dim, 1, 1, device=device)
        fake_imgs = netG(noise)
        outputs_fake = netD(fake_imgs.detach())
        d_loss_fake = criterion(outputs_fake, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizerD.step()

        optimizerG.zero_grad()
        outputs = netD(fake_imgs)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizerG.step()

        loop.set_postfix(Loss_D=d_loss.item(), Loss_G=g_loss.item())

    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        utils.save_image(fake, os.path.join(sample_dir, f"fake_samples_epoch_{epoch + 1}.png"), normalize=True)

print("Training complete!")

netG.eval()
with torch.no_grad():
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
    fake_images = netG(fixed_noise).detach().cpu()

grid = utils.make_grid(fake_images, padding=2, normalize=True)
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Generated Anime Faces")
plt.imshow(grid.permute(1, 2, 0))
plt.show()

torch.save(netG.state_dict(), "anime_generator.pth")

