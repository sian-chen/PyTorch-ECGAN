import numpy as np
from tqdm import tqdm

from utils.sample import sample_latents
from utils.losses import latent_optimise

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from pytorchcv.model_provider import get_model as ptcv_get_model


def calculate_classifier_accuracy_score(dataloader, generator, discriminator, num_evaluate, truncated_factor, prior, latent_op,
                       latent_op_step, latent_op_alpha, latent_op_beta, device, logger, eval_generated_sample=False):
    data_iter = iter(dataloader)
    batch_size = dataloader.batch_size
    disable_tqdm = device != 0
    net = ptcv_get_model('wrn40_8_cifar10', pretrained=True).to(device)

    if isinstance(generator, DataParallel) or isinstance(generator, DistributedDataParallel):
        z_dim = generator.module.z_dim
        num_classes = generator.module.num_classes
        conditional_strategy = discriminator.module.conditional_strategy
    else:
        z_dim = generator.z_dim
        num_classes = generator.num_classes
        conditional_strategy = discriminator.conditional_strategy

    total_batch = num_evaluate//batch_size

    if device == 0: logger.info("Calculate Classifier Accuracy Score....")

    all_pred_fake, all_pred_real, all_fake_labels, all_real_labels = [], [], [], []
    for batch_id in tqdm(range(total_batch), disable=disable_tqdm):
        zs, fake_labels = sample_latents(prior, batch_size, z_dim, truncated_factor, num_classes, None, device)
        if latent_op:
            zs = latent_optimise(zs, fake_labels, generator, discriminator, conditional_strategy, latent_op_step,
                                    1.0, latent_op_alpha, latent_op_beta, False, device)

        real_images, real_labels = next(data_iter)
        real_images, real_labels = real_images.to(device), real_labels.to(device)

        fake_images = generator(zs, fake_labels, evaluation=True)

        with torch.no_grad():
            pred_fake = net(fake_images).detach().cpu().numpy()
            pred_real = net(real_images).detach().cpu().numpy()

        all_pred_fake.append(pred_fake)
        all_pred_real.append(pred_real)
        all_fake_labels.append(fake_labels.cpu().numpy())
        all_real_labels.append(real_labels.cpu().numpy())

    all_pred_fake = np.concatenate(all_pred_fake, axis=0).argmax(axis=1)
    all_pred_real = np.concatenate(all_pred_real, axis=0).argmax(axis=1)
    all_fake_labels = np.concatenate(all_fake_labels)
    all_real_labels = np.concatenate(all_real_labels)

    fake_cas = (all_pred_fake == all_fake_labels).mean()
    real_cas = (all_pred_real == all_real_labels).mean()

    return real_cas, fake_cas
