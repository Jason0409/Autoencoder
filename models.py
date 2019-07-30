import torch
import torch.nn as nn

# TODO: Write any helper routines here.


class Encoder(nn.Module):
    """Encoder network to map from an RGB image to a latent feature vector."""

    def __init__(self, z_dim=64, img_size=64):
        super(Encoder, self).__init__()

        self.z_dim = z_dim
        self.hidden_layer = nn.Sequential(
            nn.Linear(3*img_size*img_size, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.output_layer(self.hidden_layer(x))
        return x


class Decoder(nn.Module):
    """Decoder network to map from a latent feature vector to an RGB image."""

    def __init__(self, z_dim=64, img_size=64):
        super(Decoder, self).__init__()

        assert img_size==64
        self.z_dim = z_dim
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=z_dim,
                out_channels=128,
                kernel_size=7,
                stride=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Tanh(),
        )


    def forward(self, x):
        x = x.view(x.size()[0], self.z_dim, 1, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
