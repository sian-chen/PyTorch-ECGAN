import numpy as np
from tqdm import tqdm

from utils.sample import sample_latents
from utils.losses import latent_optimise

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel


def calculate_discriminator_classification_accuracy(dataloader, generator, discriminator, num_evaluate, truncated_factor, prior, latent_op,
                       latent_op_step, latent_op_alpha, latent_op_beta, device, logger, eval_generated_sample=False):
    data_iter = iter(dataloader)
    batch_size = dataloader.batch_size
    disable_tqdm = device != 0

    if isinstance(generator, DataParallel) or isinstance(generator, DistributedDataParallel):
        conditional_strategy = discriminator.module.conditional_strategy
    else:
        conditional_strategy = discriminator.conditional_strategy

    total_batch = num_evaluate//batch_size

    if device == 0: logger.info("Calculate Discriminator Classification Accuracy....")

    all_pred_real, all_real_labels = [], []
    for batch_id in tqdm(range(total_batch), disable=disable_tqdm):
        real_images, real_labels = next(data_iter)
        real_images, real_labels = real_images.to(device), real_labels.to(device)

        with torch.no_grad():
            if conditional_strategy == "ACGAN":
                cls_out_real, _ = discriminator(real_images, real_labels)
            elif conditional_strategy == 'ECGAN':
                cls_out_real, _, _, _, _ = discriminator(real_images, real_labels)
            pred_real = cls_out_real.detach().cpu().numpy()

        all_pred_real.append(pred_real)
        all_real_labels.append(real_labels.cpu().numpy())

    all_pred_real = np.concatenate(all_pred_real, axis=0).argmax(axis=1)
    all_real_labels = np.concatenate(all_real_labels)

    dca = (all_pred_real == all_real_labels).mean()

    return dca
