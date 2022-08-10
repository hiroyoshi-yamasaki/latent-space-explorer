import numpy as np
import torch
from tqdm import tqdm

from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets
from torchvision import transforms

from models.model import Model
from utils import squeeze_components, unsqueeze_components
from params import *


def get_loader(batch_size):

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),    # mean
                                                         (0.3081,))])  # std

    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)
    return loader


class AutoEncoder(nn.Module, Model):

    def __init__(self):
        nn.Module.__init__(self)
        Model.__init__(self)

        #
        self.compressed = None

        #
        self.encoder = nn.Sequential(nn.Linear(IMG_SIZE * IMG_SIZE, LAYER_1),
                                     nn.ReLU(),
                                     nn.Linear(LAYER_1, LAYER_2),
                                     nn.ReLU(),
                                     nn.Linear(LAYER_2, N_COLS * N_ROWS),
                                     nn.ReLU())

        #
        self.decoder = nn.Sequential(nn.Linear(N_COLS * N_ROWS, LAYER_2),
                                     nn.ReLU(),
                                     nn.Linear(LAYER_2, LAYER_1),
                                     nn.ReLU(),
                                     nn.Linear(LAYER_1, IMG_SIZE * IMG_SIZE),
                                     nn.Sigmoid())

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded

    def encode(self, x):
        # with torch.no_grad: TODO: apply no_grad
        return self.encoder(x)

    def decode(self, latent):
        # with torch.no_grad:
        return self.decoder(latent)

    def fit(self, epochs=100, lr=LEARNING_RATE, weight_decay=DECAY, batch_size=BATCH_SIZE):

        avr_loss = []
        loader = get_loader(batch_size=batch_size)
        optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(epochs):

            print(f"Training {epoch}th epoch")

            losses = []
            for image, _ in tqdm(loader):

                #
                image = image.reshape(-1, IMG_SIZE * IMG_SIZE)
                reconstruction = self(image)

                #
                loss_func = nn.MSELoss()
                loss = loss_func(reconstruction, image)

                #
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #
                losses.append(loss.item())

            #
            loss = np.array(losses).mean()
            avr_loss.append(loss)
            print(f"Average loss: {loss}")

        self._save_compressed(loader)

    def _save_compressed(self, loader, max_n=10000):

        compressed = []
        for i, (image, _) in enumerate(tqdm(loader)):
            compressed.append(self.encode(image.reshape(-1)).detach().numpy())

            if i > max_n:
                break

        self.compressed = np.array(compressed)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self._save_compressed(get_loader(batch_size=1))

    def sample(self):
        # if not self.compressed:  TODO: check if compressed is there
        #    raise ValueError("Compressed must be computed first")

        n_samples = self.compressed.shape[0]
        idx = np.random.choice(n_samples, size=1)
        sample = self.generate(self.compressed[idx])

        #
        compressed = squeeze_components(components=self.compressed[idx].reshape(-1),  # selected component
                                        compressed=self.compressed)                   # for calculating min-max

        return sample.reshape(28, 28), compressed

    def generate(self, components):
        components = unsqueeze_components(components=components, compressed=self.compressed)
        return self.decode(torch.Tensor(components)).detach().numpy()

    def get_total_params(self):
        return sum(param.numel() for param in self.parameters())


"""
class ConvAutoEncoder(nn.Module, Model):

    def __init__(self, load=False):
        nn.Module.__init__(self)
        Model.__init__(self)

        #
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=CHANNELS_1,
                               kernel_size=KERNEL_1, stride=2, padding=5)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=CHANNELS_2,
                               kernel_size=KERNEL_2, stride=2, padding=4)

        self.enc1 = nn.Linear(HIDDEN_1 * HIDDEN_1 * CHANNELS_2, HIDDEN_2)
        self.enc2 = nn.Linear(HIDDEN_2, N_COLS * N_ROWS)

        #
        self.dec1 = nn.Linear(N_COLS * N_ROWS, HIDDEN_2)
        self.dec2 = nn.Linear(HIDDEN_2, HIDDEN_1 * HIDDEN_1 * CHANNELS_2)
        self.deconv1 = nn.ConvTranspose2d(in_channels=CHANNELS_2, out_channels=CHANNELS_1,
                                          kernel_size=KERNEL_2, stride=2, padding=4)
        self.deconv2 = nn.ConvTranspose2d(in_channels=CHANNELS_2, out_channels=1,
                                          kernel_size=KERNEL_1, stride=2, padding=5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = x.view(-1, d * d * c)
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))

        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x)).view(-1, c, d, d)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        # print(x.shape)
        return x
"""