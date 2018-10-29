#!/usr/bin/env python
#
# COMP4680/8650: ADVANCED TOPICS IN STATISTICAL MACHINE LEARNING
# ASSIGNMENT 6
#
# DO NOT MODIFY ANYTHING IN THIS FILE
#

import unittest

import torch
from torch.autograd import Variable

import models


# test encoder
class TestEncoder(unittest.TestCase):

    def test_size(self):
        """Test that the Encoder produces the correct size output."""
        batch_size = 32
        image_size = 64
        latent_size = 128
        
        E = models.Encoder(latent_size, image_size)
        tensor_in = Variable(2.0 * (torch.rand((batch_size, 3, image_size, image_size)) - 0.5))
        tensor_out = E.forward(tensor_in)
        self.assertEqual(tensor_out.size(), torch.Size([batch_size, latent_size]))


    def test_layers(self):
        """Test that the Encoder produces the correct size output for each layer."""
        batch_size = 32
        image_size = 64
        latent_size = 128

        E = models.Encoder(latent_size, image_size)
        tensor = Variable(2.0 * (torch.rand((batch_size, 3, image_size, image_size)) - 0.5))
        tensor = tensor.view(tensor.size()[0], -1)
        tensor = E.hidden_layer(tensor)
        self.assertEqual(tensor.size(), torch.Size([batch_size, latent_size]))
        tensor = E.output_layer(tensor)
        self.assertEqual(tensor.size(), torch.Size([batch_size, latent_size]))


# test decoder
class TestDecoder(unittest.TestCase):

    def test_size(self):
        """Test that the Decoder produced the correct size output."""
        batch_size = 32
        image_size = 64
        latent_size = 128

        D = models.Decoder(latent_size, image_size)
        tensor_in = Variable(2.0 * (torch.rand((batch_size, latent_size, 1, 1)) - 0.5))
        tensor_out = D.forward(tensor_in)
        self.assertEqual(tensor_out.size(), torch.Size([batch_size, 3, image_size, image_size]))


    def test_layers(self):
        """Test that the Decoder produced the correct size output for each layer."""
        batch_size = 32
        image_size = 64
        latent_size = 128

        D = models.Decoder(latent_size, image_size)
        tensor = Variable(2.0 * (torch.rand((batch_size, latent_size, 1, 1)) - 0.5))
        tensor = D.layer1.forward(tensor)
        self.assertEqual(tensor.size(), torch.Size([batch_size, 128, 7, 7]))
        tensor = D.layer2.forward(tensor)
        self.assertEqual(tensor.size(), torch.Size([batch_size, 64, 15, 15]))
        tensor = D.layer3.forward(tensor)
        self.assertEqual(tensor.size(), torch.Size([batch_size, 64, 31, 31]))
        tensor = D.layer4.forward(tensor)
        self.assertEqual(tensor.size(), torch.Size([batch_size, 32, 63, 63]))
        tensor = D.layer5.forward(tensor)
        self.assertEqual(tensor.size(), torch.Size([batch_size, 3, image_size, image_size]))


# main
if __name__ == '__main__':
    unittest.main()
