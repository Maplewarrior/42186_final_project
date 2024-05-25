import torchvision
import matplotlib.pyplot as plt
import torch

def save_images(images, path, show=False, title=None, nrow=8):
    """
    Function for displaying and saving diffusion samples.
    """
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    if title is not None:
        plt.title(title)
    plt.imshow(ndarr)
    plt.axis('off')
    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()

def denormalize(x):
    """
    Function that maps an input from range [-1, 1] --> [0, 255]
        - Used for displaying diffusion samples
    """
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x