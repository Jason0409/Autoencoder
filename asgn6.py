import os
import argparse
import numpy as np
import time

import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import models


# --- utility code -------------------------------------------------------------

# Run on GPU if available. Set to False if you want to force the models
# to stay on the CPU even if cuda is available.
RUN_ON_GPU = torch.cuda.is_available()

# Set the random seed for reproducibility
SEED = 4680

np.random.seed(SEED)
torch.manual_seed(SEED)
if RUN_ON_GPU:
    torch.cuda.manual_seed(SEED)


def np_to_var(x):
    """Converts numpy to variable."""
    if RUN_ON_GPU:
        x = x.cuda()
    return Variable(x)


def var_to_np(x):
    """Converts variable to numpy."""
    if RUN_ON_GPU:
        x = x.cpu()
    return x.data.numpy()


def get_data_loader(image_size=64, batch_size=16):
    """Creates training data loader."""
    transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    train_dataset = datasets.ImageFolder('./data/', transform)
    return DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# --- training code -------------------------------------------------------

def create_models(image_size=64, latent_size=128):
    """Creates encoder and decoder and moves them to the GPU if requested."""

    E = models.Encoder(latent_size, image_size)
    D = models.Decoder(latent_size, image_size)

    if RUN_ON_GPU:
        print('Moving models to GPU.')
        E.cuda()
        D.cuda()
    else:
        print('Keeping models on CPU.')

    return E, D


def train_models(E, D, image_size=64, latent_size=128, batch_size=32, num_epochs=500,
                 log_interval=10, save_interval=100, out_dir='samples'):
    """Train models."""

    # create optimizers
    e_optimizer = optim.Adam(E.parameters(), 1.0e-3, [0.5, 0.999])
    d_optimizer = optim.Adam(D.parameters(), 2.0e-3, [0.5, 0.999])

    # create output directory for image samples
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dataloader = get_data_loader(image_size=image_size, batch_size=batch_size)
    total_train_iters = num_epochs * len(dataloader)

    loss_history = []
    print('Started training at {}'.format(time.asctime(time.localtime(time.time()))))

    # train for num_epochs
    for epoch in range(num_epochs):
        for iteration, batch in enumerate(dataloader, epoch * len(dataloader) + 1):

            real_images, _ = batch
            noisy_images = torch.clamp(torch.add(real_images, 2.0 * (torch.rand(real_images.size()) - 0.5)), min=-1.0, max=1.0)
            real_images = np_to_var(real_images)
            noisy_images = np_to_var(noisy_images)

            e_optimizer.zero_grad()
            d_optimizer.zero_grad()
            out_images = D.forward(E.forward(noisy_images))

            # compute reconstruction loss and update parameters
            loss = torch.mean((real_images - out_images).abs())
            loss.backward()
            e_optimizer.step()
            d_optimizer.step()

            # print the log info
            if iteration % log_interval == 0:
                print('Iteration [{:6d}/{:6d}] | loss: {:.4f}'.format(
                    iteration, total_train_iters, loss.data.item()))

            # keep track of loss for plotting and saving
            loss_history.append(loss.data.item())

            # save the generated samples and loss
            if iteration % save_interval == 0:
                path = os.path.join(out_dir, 'sample-{:06d}.png'.format(iteration))
                torchvision.utils.save_image(torch.cat((real_images, noisy_images, out_images)),
                                             path, nrow=real_images.size()[0], normalize=True)
                print('Saved {}'.format(path))

                # save the loss history
                with open('loss.txt', 'wt') as file:
                    file.write('\n'.join(['{}'.format(loss) for loss in loss_history]))
                    file.write('\n')

                # generate and save some novel images
                noise = np_to_var(2.0 * (torch.rand((batch_size, latent_size, 1, 1)) - 0.5))
                novel_images = D.forward(noise)
                path = os.path.join(out_dir, 'novel-{:06d}.png'.format(iteration))
                torchvision.utils.save_image(novel_images, path, normalize=True)
                print('Saved {}'.format(path))


# --- main ----------------------------------------------------------------

def main():
    # set up commandline arguments
    parser = argparse.ArgumentParser(description='COMP4680/8650 Assignment 6')
    parser.add_argument('--image-size', type=int, default=64, metavar='N',
                        help='image dimensions (width and height) (default: 64)')
    parser.add_argument('--latent-size', type=int, default=128, metavar='N',
                        help='latent representation size (default: 128)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='iterations to wait between printing progress (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100, metavar='N',
                        help='iterations to wait between saving samples (default: 100)')
    parser.add_argument('--output-dir', type=str, default='samples', metavar='DIR',
                        help='output directory for image samples')

    args = parser.parse_args()

    # create and train models
    E, D = create_models(args.image_size, args.latent_size)
    train_models(E, D, image_size=args.image_size, latent_size=args.latent_size,
                 batch_size=args.batch_size, num_epochs=args.epochs,
                 log_interval=args.log_interval, save_interval=args.save_interval,
                 out_dir=args.output_dir)


if __name__ == '__main__':
    main()
