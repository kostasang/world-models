from typing import Callable
import torch, wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.plotter import make_collage

def train_v_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_functions,
    epochs: int,
    evaluation_steps: int,
    plotting_epochs: int,
    best_model_path: str
):
    """
    Implements training loop for the V-Model
    """
    # obtain the model's device ID
    #device = next(model.parameters()).device
    model.train()
    global_step = 0
    best_score = 999999
    for epoch in tqdm(range(epochs)):
        training_step = 0
        sum_loss = 0
        for batch in train_dataloader:    
            optimizer.zero_grad()
            y_pred = model(batch)
            loss = 0
            for loss_func in loss_functions:
                loss += loss_func(y_pred, batch)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            training_step += 1
            global_step += 1
            if evaluation_steps > 0 and training_step % evaluation_steps == 0:
                validation_loss = evaluate_model(
                    model=model,
                    validation_loader=validation_dataloader,
                    loss_functions=loss_functions
                )
                wandb.log(
                    data={
                        'training_loss': (sum_loss/training_step),
                        'validation_loss': validation_loss,
                    },
                    commit=True,
                    step=global_step
                )
                if validation_loss < best_score:
                    best_score = validation_loss
                    torch.save(model.state_dict(), best_model_path)
        if epoch % plotting_epochs == 0:
            image_batch = next(iter(validation_dataloader))
            image_batch = image_batch[:50]
            predicted_images = model(image_batch)
            make_collage(
                real_images=image_batch.cpu().detach().permute((0,2,3,1)).numpy(),
                reconstructed_images=predicted_images.cpu().detach().permute((0,2,3,1)).numpy(),
                epoch=epoch
            )


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    validation_loader: DataLoader,
    loss_functions
) -> float:
    """
    Evaluates model on evaluation dataloader and returns loss
    """
    total_loss = 0 
    model.eval()
    for idx, batch in enumerate(validation_loader):
        y_pred = model(batch)
        loss = 0
        for loss_func in loss_functions:
            loss += loss_func(y_pred, batch)
        total_loss += loss.item()
    model.train()
    return total_loss / (idx+1)
