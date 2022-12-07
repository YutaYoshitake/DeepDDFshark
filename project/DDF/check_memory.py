import pdb
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from parser import *
from train_pl import DDF



class voxelsdf(pl.LightningModule):

    def __init__(self, latent_size):
        super().__init__()

        self.cov_inp_size = 512
        self.decoder_fc = nn.Sequential(
                nn.Linear(latent_size, self.cov_inp_size), nn.LeakyReLU(0.2)
                )
        self.decoder_cov = nn.Sequential(
                nn.ConvTranspose3d(self.cov_inp_size, 256, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose3d(256, 128, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose3d(128, 64, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose3d(64, 32, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose3d(32, 16, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose3d(16, 8, 4, 2, 1), 
                nn.Tanh(),
                )

    def forward(self, inp):
        x = self.decoder_fc(inp)
        x = x.view(-1, self.cov_inp_size, 1, 1, 1)
        x = self.decoder_cov(x)
        return x



class deepsdf(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(deepsdf, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x



batch_size = 1

# # Get voxelsdf memory.
# model = deepsdf(
#         256,
#         dims = [512, 512, 512, 512, 512, 512, 512, 512],
#         dropout = [0, 1, 2, 3, 4, 5, 6, 7],
#         dropout_prob = 0.2,
#         norm_layers = [0, 1, 2, 3, 4, 5, 6, 7],
#         latent_in = [4],
#         weight_norm = True,
#         xyz_in_all = False,
#         use_tanh = False,
#         latent_dropout = False,
#         )
# summary(
#     model,
#     input_size=(batch_size, 256+3),
#     col_names=["output_size", "num_params"],
# )



# Get voxelsdf memory.
model = voxelsdf(256)
summary(
    model,
    input_size=(batch_size, 256),
    col_names=["output_size", "num_params"],
)



# Get voxelsdf memory.
# args = get_args()
# args.gpu_num = torch.cuda.device_count() # log used gpu num.
# args.ddf_H = args.H

# model = DDF(args)
# summary(
#     model,
#     input_size=(batch_size, 256+3+3),
#     # col_names=["output_size", "num_params"],
# )
# model(torch.ones(1, 256+3+3))



import pdb; pdb.set_trace()