from typing import List

import torch
from allennlp.common import FromParams
from torch import nn
from torch.nn import Parameter
from torch.nn.functional import mse_loss


class DenseVAE(nn.Module, FromParams):

    def __init__(self,
                 input_dim: int = 1024,
                 embedding_dim: int = 64,
                 hidden_dims: List[int] = [512, 256, 128],
                 negative_slope=0.1,
                 tied_weights: bool = False) -> None:
        super(DenseVAE, self).__init__()

        self.input_dim = input_dim

        # Build encoder

        encoder_modules = []

        for h_dim in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Linear(in_features=input_dim, out_features=h_dim),
                    nn.LeakyReLU(negative_slope=negative_slope))
            )
            input_dim = h_dim

        self.dense_mu = nn.Linear(hidden_dims[-1], embedding_dim)
        self.dense_var = nn.Linear(hidden_dims[-1], embedding_dim)

        # Build Decoder
        decoder_modules = []

        hidden_dims.reverse()

        input_dim = embedding_dim
        for h_dim in hidden_dims:
            decoder_modules.append(
                nn.Sequential(
                    nn.Linear(in_features=input_dim, out_features=h_dim),
                    nn.LeakyReLU(negative_slope=negative_slope))
            )
            input_dim = h_dim


        # Final layer to get back to the input size.
        decoder_modules.append(nn.Linear(in_features=input_dim, out_features=self.input_dim))

        # Tie the weights. Don't tie the first decoder weight as this is split for mu and var on the encoder.
        if tied_weights:
            for enc, dec in zip(reversed(list(encoder_modules)), list(decoder_modules)[1:]):
                dec[0].weight = Parameter(enc[0].weight.t(), requires_grad=True)

        self._encoder = nn.Sequential(*encoder_modules)
        self._decoder = nn.Sequential(*decoder_modules)

    def encode(self, x):
        x = self._encoder(x)
        return self.dense_mu(x), self.dense_var(x)


    def decode(self, z):
        return self._decoder(z)


    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        # logger.info(f"Autoencoder Forward {x.size()}")
        mu, logvar = self.encode(x)
        z = self._reparameterize(mu, logvar)
        y = self.decode(z)
        return y, x, mu, logvar

    def loss_function(self, input, recons, mu, logvar, kld_weighting: int = 1.0):
        mse = mse_loss(recons, input)
        kld = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) * kld_weighting
        return mse + kld
