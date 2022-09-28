from torch.utils.data import DataLoader
from models.v_model import VModel
from datasets.image_dataset import ImageDataset
from training.v_trainer import train_v_model
from utils.configs import load_configurations


if __name__ == "__main__":

    configs = load_configurations(path='configs/training_vmodel.yaml')

    train_dataset = ImageDataset(storage_folder=configs.data.train_set)
    val_dataset = ImageDataset(storage_folder=configs.data.validation_set)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=configs.training.train_batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=configs.training.val_batch_size, shuffle=False)

    model = VModel(n_z=configs.model.n_z)

    train_v_model()