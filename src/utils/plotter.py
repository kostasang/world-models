import numpy as np
import matplotlib.pyplot as plt
import wandb

def make_collage(
    real_images: np.ndarray,
    reconstructed_images: np.ndarray,
    epoch: int
):
    """
    Plots a collage of reconstructed images coming from the V-model 
    along with the real images
    """
    fig = plt.figure(figsize=(10,10))
    for idx in range(real_images.shape[0]):
        plt.subplot(10, 10, 2*idx+1)
        plt.imshow(real_images[idx])
        plt.axis('off')
        plt.subplot(10,10,2*idx+2)
        plt.imshow(reconstructed_images[idx])
        plt.axis('off')
    plt.tight_layout()
    wandb.log({f'results_epoch_{epoch}': fig})
