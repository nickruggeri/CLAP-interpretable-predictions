from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from .neural_net import CLAPDecoder, CLAPEncoderBackbone, LinearClassifier


class CLAP(nn.Module):
    """CLAP architecture.
    CLAP is composed of two partially overlapping VAEs: a prediction VAE and a concept learning VAE. These two VAEs
    share the same decoder. The encoder is shared partially:
    - a shared backbone structure maps images x to the style latent features of both VAEs
    - another encoder maps x to the core latent features of the prediction VAE
    - another encoder maps x and y to the core latent features of the concept learning VAE
    All these encoders have a very similar backbone structure, with linear layers added to adjust for the different
    input or output dimensions.
    """

    def __init__(
        self,
        n_channels: int,
        in_dim: int,
        z_style_dim: int,
        z_core_dim: int,
        n_outputs: int,
        intermediate_size=256,
    ) -> None:
        """
        :param n_channels: number of channels in the input images
        :param in_dim: dimension of the input images, which have then shape (batch_size, n_channels, in_dim, in_dim)
        :param z_style_dim: dimension of the style latent space
        :param z_core_dim: dimension of the core latent space
        :param n_outputs: number of binary labels in the supervision y, which has shape (batch_size, n_outputs)
        :param intermediate_size: intermediate size, output of the encoder backbone shared by the prediction and
        concept learning VAEs
        """
        super().__init__()

        self.n_channels = n_channels
        self.in_dim = in_dim
        self.z_style_dim = z_style_dim
        self.z_core_dim = z_core_dim
        self.n_outputs = n_outputs
        self.intermediate_size = intermediate_size

        (
            self.decoder,  # shared decoder
            self.cl_vae,  # concept learning VAE
            self.pred_vae,  # prediction VAE
        ) = self.construct_clap_parts()

    def construct_clap_parts(self) -> Tuple[nn.Module, nn.Module, nn.Module]:
        shared_encoder_backbone = CLAPEncoderBackbone(self.n_channels, self.in_dim)
        shared_style_mean_layer = nn.Linear(self.intermediate_size, self.z_style_dim)
        shared_style_log_var_layer = nn.Linear(self.intermediate_size, self.z_style_dim)

        cl_core_encoder_backbone = nn.Sequential(
            CLAPEncoderBackbone(self.n_channels, self.in_dim, self.intermediate_size),
            nn.Linear(self.intermediate_size, 32),
            nn.ReLU(nn.ReLU(inplace=True)),
            nn.Dropout(p=0.05),
        )
        cl_core_mean_layer = nn.Linear(32 + self.n_outputs, self.z_core_dim)
        cl_core_log_var_layer = nn.Linear(32 + self.n_outputs, self.z_core_dim)

        decoder = CLAPDecoder(self.n_channels, self.z_core_dim, self.z_style_dim)

        # prediction (P) part of CLAP
        pred_vae = PredictionVAE(
            encoder_backbone=shared_encoder_backbone,
            style_mean_layer=shared_style_mean_layer,
            style_log_var_layer=shared_style_log_var_layer,
            core_mean_layer=nn.Linear(self.intermediate_size, self.z_core_dim),
            core_log_var_layer=nn.Linear(self.intermediate_size, self.z_core_dim),
            decoder=decoder,
            predictor=LinearClassifier(self.z_core_dim, self.n_outputs),
        )

        # concept learning (CL) part of CLAP
        cl_vae = ConceptLearningVAE(
            style_encoder_backbone=shared_encoder_backbone,
            core_encoder_backbone=cl_core_encoder_backbone,
            style_mean_layer=shared_style_mean_layer,
            style_log_var_layer=shared_style_log_var_layer,
            core_mean_layer=cl_core_mean_layer,
            core_log_var_layer=cl_core_log_var_layer,
            decoder=decoder,
            n_outputs=self.n_outputs,
        )

        return decoder, cl_vae, pred_vae

    def get_decoder_first_linear_layer(self) -> torch.Tensor:
        return self.decoder.get_first_linear_layer()

    def get_prediction_weights(self) -> torch.Tensor:
        return self.pred_vae.predictor.get_weights()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        pred_out = self.pred_vae(x)
        cl_out = self.cl_vae(x, y)

        return {
            "pred": pred_out,
            "cl": cl_out,
        }


class ConceptLearningVAE(nn.Module):
    def __init__(
        self,
        style_encoder_backbone: nn.Module,
        core_encoder_backbone: nn.Module,
        style_mean_layer: nn.Module,
        style_log_var_layer: nn.Module,
        core_mean_layer: nn.Module,
        core_log_var_layer: nn.Module,
        decoder: nn.Module,
        n_outputs: int,
    ) -> None:
        super().__init__()
        self.n_outputs = n_outputs

        self.style_encoder_backbone = style_encoder_backbone
        self.core_encoder_backbone = core_encoder_backbone

        # the following four layers maps the intermediate encoding of
        # self.style_encoder_backbone and self.core_encoder_backbone to
        # the mean and log-variance of the core and style latent features
        self.style_mean_layer = style_mean_layer
        self.style_log_var_layer = style_log_var_layer
        self.core_mean_layer = core_mean_layer
        self.core_log_var_layer = core_log_var_layer

        self.decoder = decoder

        (
            self.core_prior_mean,
            self.core_prior_log_var,
        ) = self._register_learnable_priors()

    def _register_learnable_priors(self) -> Tuple[nn.Module, nn.Module]:
        """Create the neural network layers that represent the conditional prior p(z | y) for the core features."""
        z_core_dim = next(self.core_mean_layer.parameters()).shape[0]

        mean_cl = nn.Linear(self.n_outputs, z_core_dim)
        log_var_cl = nn.Linear(self.n_outputs, z_core_dim)
        return mean_cl, log_var_cl

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        # ENCODING
        # style features
        encoded_x_style = self.style_encoder_backbone(x)
        mean_style = self.style_mean_layer(encoded_x_style)
        log_var_style = self.style_log_var_layer(encoded_x_style)
        z_style = Normal(loc=mean_style, scale=torch.exp(0.5 * log_var_style)).rsample()

        # core features
        encoded_x_core = self.core_encoder_backbone(x)
        if len(y.shape) == 1:
            encoded_xy = torch.cat([encoded_x_core, y.unsqueeze(dim=-1)], dim=-1)
        else:
            encoded_xy = torch.cat([encoded_x_core, y], dim=-1)
        mean_core = self.core_mean_layer(encoded_xy)
        log_var_core = self.core_log_var_layer(encoded_xy)
        z_core = Normal(loc=mean_core, scale=torch.exp(0.5 * log_var_core)).rsample()

        # PRIOR computation
        float_y = y.type(torch.float)
        prior_mean_core = self.core_prior_mean(float_y)
        prior_log_var_core = self.core_prior_log_var(float_y)

        # RECONSTRUCTION of the input image
        z = torch.cat([z_core, z_style], dim=-1)
        reconstruction = self.decoder(z)

        return {
            "mean_core": mean_core,
            "log_var_core": log_var_core,
            "z_core": z_core,
            "mean_style": mean_style,
            "log_var_style": log_var_style,
            "z_style": z_style,
            "x_reconstructed": reconstruction,
            "prior_mean_core": prior_mean_core,
            "prior_log_var_core": prior_log_var_core,
        }


class PredictionVAE(nn.Module):
    def __init__(
        self,
        encoder_backbone: nn.Module,
        style_mean_layer: nn.Module,
        style_log_var_layer: nn.Module,
        core_mean_layer: nn.Module,
        core_log_var_layer: nn.Module,
        decoder: nn.Module,
        predictor: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder_backbone = encoder_backbone

        # the following four layers maps the intermediate encoding of
        # self.encoder_backbone to the mean and log-variance of the
        # core and style latent spaces
        self.style_mean_layer = style_mean_layer
        self.style_log_var_layer = style_log_var_layer
        self.core_mean_layer = core_mean_layer
        self.core_log_var_layer = core_log_var_layer

        self.decoder = decoder
        self.predictor = predictor

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # ENCODING the input into mean and style features
        encoded = self.encoder_backbone(x)

        mean_style = self.style_mean_layer(encoded)
        log_var_style = self.style_log_var_layer(encoded)
        z_style = Normal(loc=mean_style, scale=torch.exp(0.5 * log_var_style)).rsample()

        mean_core = self.core_mean_layer(encoded)
        log_var_core = self.core_log_var_layer(encoded)
        z_core = Normal(loc=mean_core, scale=torch.exp(0.5 * log_var_core)).rsample()

        # PREDICTION from core feature
        # Change the self.training attribute calling .eval() or .train() methods
        if self.training:
            y_pred = self.predictor(z_core)
        else:
            y_pred = self.predictor(mean_core)

        # RECONSTRUCTION of the input image
        z = torch.cat([z_core, z_style], dim=-1)
        reconstruction = self.decoder(z)

        return {
            "mean_core": mean_core,
            "log_var_core": log_var_core,
            "z_core": z_core,
            "mean_style": mean_style,
            "log_var_style": log_var_style,
            "z_style": z_style,
            "x_reconstructed": reconstruction,
            "y_pred": y_pred,
        }


def logits_to_labels(logit: torch.Tensor) -> torch.Tensor:
    """Convert tensor of logit values to binary labels."""
    return torch.where(logit > 0, 1, 0)
