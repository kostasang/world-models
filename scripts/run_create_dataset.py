import os, numpy as np
from tqdm import tqdm
from world_models.environment.utils import load_environment
from world_models.utils.configs import load_configurations

if __name__ == "__main__":

    configs = load_configurations(path='configs/dataset_creation.yaml')
    env = load_environment(**configs.environment)

    trajectory_collection = []
    rewards_collection = []
    actions_collection = []
    status_collection = []
    pbar = tqdm(total=configs.dataset.n_sequencies)
    while(len(trajectory_collection)< configs.dataset.n_sequencies):
        trajectory = []
        rewards = []
        actions = []
        status = []

        done = False
        state = env.reset()
        while not done:
            action = env.action_space.sample()
            trajectory.append(state)
            actions.append(action)
            state, reward, done, _ = env.step(action)
            status.append(done)
            rewards.append(reward)
        
        if len(trajectory) >= configs.dataset.min_seq_size:
            trajectory_collection.append(np.asarray(trajectory))
            rewards_collection.append(np.asarray(rewards))
            actions_collection.append(np.asarray(actions))
            status_collection.append(np.asarray(status))
            pbar.update()

    if not os.path.isdir(configs.dataset.storage_folder):
        os.mkdir(configs.dataset.storage_folder)
    np.save(configs.dataset.storage_folder+'/trajectories.npy', trajectory_collection)
    np.save(configs.dataset.storage_folder+'/rewards.npy', rewards_collection)
    np.save(configs.dataset.storage_folder+'/actions.npy', actions_collection)
    np.save(configs.dataset.storage_folder+'/status.npy', status_collection)