#!/usr/bin/env python3

from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from few_shot_learning_system import MAMLFewShotClassifier
from utils.dataset_tools import maybe_unzip_dataset

import torch

from omegaconf import OmegaConf
import hydra

from setproctitle import setproctitle
setproctitle('maml++')

@hydra.main(config_path='config.yaml', strict=True)
def main(cfg):
    # import sys
    # from IPython.core import ultratb
    # sys.excepthook = ultratb.FormattedTB(mode='Verbose',
    #     color_scheme='Linux', call_pdb=1)

    device = torch.device('cuda')
    model = MAMLFewShotClassifier(args=cfg, device=device)
    maybe_unzip_dataset(args=cfg)
    data = MetaLearningSystemDataLoader
    maml_system = ExperimentBuilder(model=model, data=data, args=cfg, device=device)
    maml_system.run_experiment()

if __name__ == '__main__':
    main()
