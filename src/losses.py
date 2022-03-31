from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def accuracy(
    y_pred: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a tensor y_pred of logits, predict the accuracy with respect to the ground truth labels y
    :param y_pred: tensor of shape (batch_size, n_labels)
    :param y: tensor of shape (batch_size, n_labels)
    :return: two tensors, one of shape (n_labels,) with the accuracy per label and one with
    the mean accuracy averaged over the batch and all the labels
    """
    # y_pred are predicted probabilities in case of binary class
    tmp = ((y_pred >= 0.5).long() == y).float()
    return torch.mean(tmp, dim=0), torch.mean(tmp)


def bernoulli_reconstruction_loss(
    reconstructed: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Reconstruction loss for images rescaled in [0, 1]. The reconstruction contains the logits of the
    pixels, with shape (batch_size, *image_dimension), while x are the original images  in [0, 1]."""
    # input reconstructions are unnormalized logits
    return (
        F.binary_cross_entropy_with_logits(reconstructed, x, reduction="sum")
        / x.size()[0]
    )


def latent_kl_divergence(
    name_z_loss: str,
    model_out_dict: Dict[str, torch.Tensor],
    z_core_dim: int,
    z_style_dim: int,
) -> torch.Tensor:
    """Various KL-divergence implementations for core and style spaces of CLAP."""
    # standard kl prior for zcore and zstyle in prediction VAE
    if name_z_loss == "prior_z_pred":
        kl_div = iso_kl_div(
            torch.cat(
                [
                    model_out_dict["pred"]["mean_core"],
                    model_out_dict["pred"]["mean_style"],
                ],
                dim=-1,
            ),
            torch.cat(
                [
                    model_out_dict["pred"]["log_var_core"],
                    model_out_dict["pred"]["log_var_style"],
                ],
                dim=-1,
            ),
        )
    # standard kl prior for zcore in prediction VAE
    elif name_z_loss == "prior_z_core_pred":
        kl_div = iso_kl_div(
            model_out_dict["pred"]["mean_core"], model_out_dict["pred"]["log_var_core"]
        )
        kl_div *= z_core_dim / (z_core_dim + z_style_dim)
    # standard kl prior for zstyle in prediction VAE
    elif name_z_loss == "prior_z_style_pred":
        kl_div = iso_kl_div(
            model_out_dict["pred"]["mean_style"],
            model_out_dict["pred"]["log_var_style"],
        )
        kl_div *= z_style_dim / (z_core_dim + z_style_dim)
    # standard kl prior for zcore and zstyle in concept learning VAE
    elif name_z_loss == "prior_z_cl":
        kl_div = iso_kl_div(
            torch.cat(
                [
                    model_out_dict["cl"]["mean_core"],
                    model_out_dict["cl"]["mean_style"],
                ],
                dim=-1,
            ),
            torch.cat(
                [
                    model_out_dict["cl"]["log_var_core"],
                    model_out_dict["cl"]["log_var_style"],
                ],
                dim=-1,
            ),
        )
    # standard kl prior for zcore in concept learning VAE
    elif name_z_loss == "prior_z_core_cl":
        kl_div = iso_kl_div(
            model_out_dict["cl"]["mean_core"], model_out_dict["cl"]["log_var_core"]
        )
        kl_div *= z_core_dim / (z_core_dim + z_style_dim)
    # standard kl prior for zstyle in concept learning VAE
    elif name_z_loss == "prior_z_style_cl":
        kl_div = iso_kl_div(
            model_out_dict["cl"]["mean_style"], model_out_dict["cl"]["log_var_style"]
        )
        kl_div *= z_core_dim / (z_core_dim + z_style_dim)
    # kl div to learned prior dependent on y in concept learning VAE
    elif name_z_loss == "prior_z_core_y_cl":
        kl_div = learned_kl_div(
            model_out_dict["cl"]["mean_core"],
            model_out_dict["cl"]["log_var_core"],
            model_out_dict["cl"]["prior_mean_core"],
            model_out_dict["cl"]["prior_log_var_core"],
        )
        kl_div *= z_core_dim / (z_core_dim + z_style_dim)
    else:
        raise NotImplementedError("unknown kl divergence.")

    return kl_div / 2


def iso_kl_div(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL-divergence between a Gaussian, specified by its mean and log-variance, and a standard Gaussian."""
    loss = 0.5 * (mean.pow(2) + log_var.exp() - log_var - 1)
    return loss.sum(1).mean()


# kl div formula from here: https://eehsan.github.io/Notes/vae.pdf
def learned_kl_div(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_log_var: torch.Tensor,
) -> torch.Tensor:
    """KL-divergence between a posterior and a prior.
    Both are Gaussian distributions, with specified mean and log-variance.
    """
    n = mean.size()[1]
    m = mean.size()[0]
    mean_cond = prior_mean.expand(m, n)
    log_var_cond = prior_log_var.expand(m, n)
    loss = 0.5 * (
        ((mean - mean_cond).pow(2) + log_var.exp()) / log_var_cond.exp()
        - log_var
        + log_var_cond
        - 1
    )
    return loss.sum(1).mean()
