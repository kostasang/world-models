# world-models

This repo constitutes an attempt to use the world-model approach described in the paper [Recurrent World Models Facilitate Policy Evolution](https://arxiv.org/abs/1809.01999) for the [DuckieTown environment](https://github.com/duckietown/gym-duckietown). The following papers where also studied:

* [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)
* [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)
* [Planning to Explore via Self-Supervised World Models](https://arxiv.org/abs/2005.05960)
* [Model-Based Imitation Learning for Urban Driving](https://arxiv.org/abs/2210.07729)

## Installation

In order to install repo run the following commands:

```
git clone https://github.com/kostasang/world-models.git
cd world-models
conda create -y --name world-models python==3.7
conda activate world-models
pip3 install -e .
```

## Scripts

Run the scripts with `python3 scripts/script_name.py`

* `run_create_dataset.py`: Creates the dataset by storing sequences of states, actions, rewards from the environment.
* `run_train_v_model.py`: Used to train the V-model.
* `run_train_m_model.py`: Used to train the M-model.

## Duckietown

In order to create dataset using DuckieTown environment, a proper Docker image must be created and run. To do this execute the `duckietown_env.sh` script. After that, create a virtual display running:

1. `Xvfb :0 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &`
2. `export DISPLAY=:0`

Afterwards, every script that regards DuckieTown can be run without problem.

## Experiment tracking

Experiment results can be found in [this](https://wandb.ai/kostasang/World-models) Wand project (currently private).




