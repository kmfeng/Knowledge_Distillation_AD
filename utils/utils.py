import torch
from torch.distributions import uniform
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
import yaml


# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def fgsm_attack(inputs, model, eps=0.1, alpha=2):
    distribution = uniform.Uniform(torch.Tensor([-eps]), torch.Tensor([eps]))
    delta = distribution.sample(inputs.shape)
    delta = torch.squeeze(delta).reshape(-1, inputs.size(1), inputs.size(2), inputs.size(3))
    delta = delta.cuda()
    inputs = inputs.cuda()
    ori_inputs = inputs
    inputs += delta
    inputs = torch.clamp(inputs, min=0, max=1)
    inputs.requires_grad = True
    outputs = model(inputs)
    model.zero_grad()
    loss_rec = torch.mean(torch.sum((outputs - ori_inputs) ** 2, dim=1))
    loss_rec.backward()
    delta = delta + alpha * inputs.grad.sign()
    delta = torch.clamp(delta, min=-eps, max=eps)
    adv_inputs = inputs + delta
    adv_inputs = torch.clamp(adv_inputs, min=0, max=1).detach_()
    return adv_inputs


def show_process_for_trainortest(input_img, recons_img, puzzled_img=None, path="./"):
    if input_img.shape[0] > 15:
        n = 15
    else:
        n = input_img.shape[0]

    channel = input_img.shape[1]

    # print("Inputs:")
    show(np.transpose(input_img[0:n].cpu().detach().numpy(), (0, 2, 3, 1)), channel=channel, path=path + "_input.png")
    # print("Puzzle Input:")
    show(np.transpose(puzzled_img[0:n].cpu().detach().numpy(), (0, 2, 3, 1)), channel=channel,
         path=path + "_puzzle_input.png")
    # print("Reconstructions:")
    show(np.transpose(recons_img[0:n].cpu().detach().numpy(), (0, 2, 3, 1)), channel=channel,
         path=path + "_reconstruction.png")


def show(image_batch, rows=1, channel=3, path="./test.png"):
    # Set Plot dimensions
    cols = np.ceil(image_batch.shape[0] / rows)
    plt.rcParams['figure.figsize'] = (0.0 + cols, 0.0 + rows)  # set default size of plots

    for i in range(image_batch.shape[0]):
        plt.subplot(rows, cols, i + 1)
        if channel != 1:
            plt.imshow(image_batch[i])
        else:
            plt.imshow(image_batch[i].reshape(image_batch.shape[-2], image_batch.shape[-2]), cmap='gray')
        plt.axis('off')
    plt.savefig(path)





