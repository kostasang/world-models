import torch, wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from world_models.datasets.sequence_dataset import SequenceDataset
from world_models.models.v_model import VModel
from world_models.models.m_model import MModel
from world_models.utils.configs import load_configurations
from world_models.training.m_trainer import train_m_model

if __name__ == "__main__":
    
    # Load configs file
    configs = load_configurations(path='configs/training_mmodel.yaml')
    wandb.init(
        project=configs.wandb.project,
        name=configs.wandb.name,
        config={**configs.model, **configs.training}
    )
    # Load trained encoder model to encode image sequencies
    vae_model = VModel(n_z=configs.model.output_dim)
    vae_model.load_state_dict(torch.load(configs.data.encoder_path))
    encoder_model = vae_model.extract_encoder()
    decoder_model = vae_model.extract_decoder()
    # Prepare datasets and dataloaders
    train_dataset = SequenceDataset(
        storage_folder=configs.data.train_set,
        visual_encoder=encoder_model
    )
    val_dataset = SequenceDataset(
        storage_folder=configs.data.validation_set,
        visual_encoder=encoder_model
    )
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
    model = MModel(**configs.model)
    optimizer = Adam(params=model.parameters(), lr=configs.training.lr)
    l2_loss = nn.MSELoss() 
    kl_loss = nn.KLDivLoss()
    # Log model to WANDB
    wandb.watch(
        models=model,
        log='all'
    )
    # Perform training
    train_m_model(
        model=model,
        train_dataloader=train_loader,
        validation_dataloader=val_loader,
        optimizer=optimizer,
        loss_functions=[l2_loss, kl_loss],
        epochs=configs.training.epochs,
        evaluation_steps=configs.training.evaluation_steps,
        plotting_epochs=configs.plotting.epochs,
        decoder_model=decoder_model,
        best_model_path=configs.saving.model_path
    )