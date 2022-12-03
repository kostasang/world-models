import torch, wandb
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from utils.plotter import make_collage

def train_m_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_functions,
    epochs: int,
    evaluation_steps: int,
    plotting_epochs: int,
    decoder_model: nn.Module,
    best_model_path: str
):
    """
    Implements training loop for the M-Model
    """
    decoder_model.eval()
    model.train()
    global_step = 0
    best_score = 999999
    for epoch in tqdm(range(epochs)):
        training_step = 0
        sum_loss = 0
        for batch in train_dataloader:
            encoded_states, actions, target_states, seq_lengths = batch    
            optimizer.zero_grad()
            state_action_features = torch.cat((encoded_states, actions), axis=-1)
            pred_states = model(state_action_features, seq_lengths)
            loss = 0
            for loss_func in loss_functions:
                loss += loss_func(pred_states, target_states)
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
            encoded_states, actions, target_states, seq_lengths = next(iter(validation_dataloader))
            state_action_features = torch.cat((encoded_states, actions), axis=-1)
            predicted_states = model(state_action_features, seq_lengths)
            filtered_target_states = []
            filtered_predicted_states = []
            for seq_length in seq_lengths:
                filtered_target_states.append(torch.flatten(target_states[:, :seq_length], start_dim=0, end_dim=1))
                filtered_predicted_states.append(torch.flatten(predicted_states[:, :seq_length], start_dim=0, end_dim=1))
            target_states = torch.vstack(filtered_target_states)[0:50]
            predicted_states = torch.vstack(filtered_predicted_states)[0:50]
            images = decoder_model(target_states)
            predicted_images = decoder_model(predicted_states)
            make_collage(
                real_images=images.cpu().detach().permute((0,2,3,1)).numpy(),
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
        encoded_states, actions, target_states, seq_lengths = batch
        state_action_features = torch.cat((encoded_states, actions), axis=-1)
        pred_states = model(state_action_features, seq_lengths)
        loss = 0
        for loss_func in loss_functions:
            loss += loss_func(pred_states, target_states)
        total_loss += loss.item()
    model.train()
    return total_loss / (idx+1)
