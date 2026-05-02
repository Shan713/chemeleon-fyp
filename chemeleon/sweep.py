from functools import partial
import copy

import wandb

from chemeleon.config import ex
from run import main


@ex.automain
def sweep(_config):
    _config = copy.deepcopy(_config)
    _config["sweep"] = True

    # login wandb
    wandb.login()

    sweep_config = {
        "method": "bayes",  # for example, grid, random, bayesian
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {
            "batch_size": {"values": [64, 128, 256]},
            "hidden_size": {"values": [64, 128, 256, 512]},
            "num_layers": {"values": [4, 6, 8]},
            "learning_rate": {"values": [1e-2, 1e-3, 1e-4, 1e-5]},
            "weight_decay": {"values": [0, 1e-2, 1e-3, 1e-4, 1e-5]},
            "optimizer": {"values": ["adam", "sgd", "adamw"]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project=_config["project_name"])

    wandb.agent(sweep_id, function=partial(main, _config), count=100)
