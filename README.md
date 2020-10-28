# MAML++ with higher model+optimizer exploration

Here we modify the official
[MAML++](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch)
code to use [higher](https://github.com/facebookresearch/higher)
for the model and optimizer so we can ablate across them.
We also add [hydra](https://hydra.cc/) for experiment management.

+ All of our model definitions are in [./models.py](./models.py)
+ The launch command for running the ablation on our cluster
  is [./launch-all.py](./launch-all.py)
+ The model monkey-patching and differentiable optimizer with
  `higher` all happens in
  [./few_shot_learning_system.py](./few_shot_learning_system.py)
+ Our plotting code is available in
  [./nbs/2019.09.14.plot.ipynb](./nbs/2019.09.14.plot.ipynb)
