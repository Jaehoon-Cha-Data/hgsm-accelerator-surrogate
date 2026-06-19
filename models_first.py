#!/usr/bin/env python3

import torch
import torch.nn as nn

import math
import types


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        module.weight.data = module.weight.data / 1.e2
        module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm1d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()


class Mixer(nn.Module):
    def __init__(self, args):
        super(Mixer, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.mixer = nn.Sequential(
            nn.Linear(self.s, self.hidden_dim, bias=self.bias),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.z * self.n_layers, bias=self.bias),
            nn.Unflatten(1, (self.n_layers, self.z))
        )

    def forward(self, x):
        x = x + torch.randn_like(x) * 0.01
        z = self.mixer(x)
        w = torch.stack([z[:, i] for i in range(self.n_layers)])
        return w

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.generator = nn.Sequential(
            nn.Linear(self.z, self.hidden_dim, bias=self.bias),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.in_feature * self.out_feature + self.out_feature, bias=self.bias)
        )

    def forward(self, x):
        x = x + torch.randn_like(x) * 0.01
        z = self.generator(x)
        w, b = torch.split(z, [self.in_feature * self.out_feature, self.out_feature], dim=-1)
        w = w.view(-1, self.in_feature, self.out_feature)
        b = b.view(1, 1, -1, self.out_feature)
        return w, b


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.discriminator = nn.Sequential(
            nn.Linear(self.z, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x):
        z = self.discriminator(x)
        return z


class FourierFeatureEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, scale=10.0):
        super(FourierFeatureEmbedding, self).__init__()

        self.B = torch.randn((embed_dim, input_dim)) * scale

    def forward(self, x):

        x_proj = 2 * math.pi * x @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class HypyerNet(nn.Module):
    def __init__(self, params):
        super(HypyerNet, self).__init__()
        self.layer0 = Generator(types.SimpleNamespace(**params['in_args']))
        self.layer1 = Generator(types.SimpleNamespace(**params['share_args']))

        self.layerd0 = Generator(types.SimpleNamespace(**params['d0_args']))
        self.layerd1 = Generator(types.SimpleNamespace(**params['d1_args']))


    def get_weights(self, z):
        w0, b0 = self.layer0(z[0])
        w1, b1 = self.layer1(z[1])
        wd0, bd0 = self.layerd0(z[2])
        wd1, bd1 = self.layerd1(z[3])
        return w0, b0, w1, b1, wd0, bd0, wd1, bd1


    def dense_block(self, x, w, b, act = nn.LeakyReLU):
        linear_out = torch.einsum('bpmi,mij->bpmj', x, w) + b
        out = act()(linear_out)
        return out

    def forward(self, z, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(2).repeat(1, 1, z.size()[1], 1)
        w0, b0, w1, b1, wd0, bd0, wd1, bd1 = self.get_weights(z)

        y0 = self.dense_block(x, w0, b0)
        y1 = self.dense_block(y0, w1, b1)

        y_d0 = self.dense_block(y1, wd0, bd0)
        y_d1 = self.dense_block(y_d0, wd1, bd1, act = nn.Identity)

        y_d1[:, :, :, 0] = nn.ReLU()(y_d1[:, :, :, 0])
        y = y_d1.permute(2, 0, 1, 3)

        return y

class QNN(nn.Module):
    def __init__(self, input_dim):
        super(QNN, self).__init__()


        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 11)

    def forward(self, x):

        x = nn.LeakyReLU()(self.fc1(x))
        x = nn.LeakyReLU()(self.fc2(x))
        x = nn.LeakyReLU()(self.fc3(x))
        x = self.fc4(x)
        return x
