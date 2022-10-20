import wandb
import torch.nn as nn

from torch.utils.data import DataLoader
from models.v_model import VModel
from datasets.image_dataset import ImageDataset
from training.v_trainer import train_v_model
from utils.configs import load_configurations
from torch.optim import Adam

if __name__ == "__main__":

    # Load configs file
    configs = load_configurations(path='configs/training_vmodel.yaml')
    wandb.init(
        project=configs.wandb.project,
        config={**configs.model, **configs.training}
    )
    # Prepare datasets and dataloaders
    train_dataset = ImageDataset(storage_folder=configs.data.train_set)
    val_dataset = ImageDataset(storage_folder=configs.data.validation_set)
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=configs.training.train_batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=configs.training.val_batch_size, 
        shuffle=False
    )
    # Prepare training ingridietns
    model = VModel(n_z=configs.model.n_z)
    optimizer = Adam(params=model.parameters(), lr=configs.training.lr)
    l2_loss = nn.MSELoss() 
    kl_loss = nn.KLDivLoss()
    # Log model to WANDB
    wandb.watch(
        models=model,
        log='all'
    )
    # Perform training
    train_v_model(
        model=model,
        train_dataloader=train_loader,
        validation_dataloader=val_loader,
        optimizer=optimizer,
        loss_functions=[l2_loss, kl_loss],
        epochs=configs.training.epochs,
        evaluation_steps=configs.training.evaluation_steps,
        plotting_epochs=configs.plotting.epochs,
        best_model_path=configs.saving.model_path
    )