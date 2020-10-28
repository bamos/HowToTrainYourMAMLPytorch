import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import hydra
import higher
import models

import copy

def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng

class MAMLFewShotClassifier(nn.Module):
    def __init__(self, device, args):
        """
        Initializes a MAML few shot learning system
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(MAMLFewShotClassifier, self).__init__()
        self.args = args
        self.device = device
        self.batch_size = args.batch_size
        # self.use_cuda = args.use_cuda
        self.current_epoch = 0

        if 'omniglot' in self.args.dataset.name:
            self.im_shape = (2, 1, 28, 28)
        elif 'imagenet' in self.args.dataset.name:
            self.im_shape = (2, 3, 84, 84)
        else:
            import ipdb; ipdb.set_trace()

        self.rng = set_torch_seed(seed=args.seed)
        kwargs = dict(
            num_classes=args.num_classes_per_set,
            im_shape=self.im_shape,
        )
        if args.net == 'vgg':
            self.classifier = models.VGGNet(
                norm_layer='batch_norm',
                cnn_num_filters=64,
                num_stages=4,
                conv_padding=True,
                max_pooling=True,
                **kwargs
            )
        elif 'resnet' in args.net:
            if args.net == 'resnet-4':
                kwargs['layers'] = [1,1,1,1]
            elif args.net == 'resnet-8':
                kwargs['layers'] = [2,2,2,2]
            elif args.net == 'resnet-12':
                kwargs['layers'] = [3,3,3,3]
            else:
                assert False

            self.classifier = models.ResNet(**kwargs)
        elif 'densenet' in args.net:
            if args.net == 'densenet-8':
                kwargs['block_config'] = [2]*4
            elif args.net == 'densenet-12':
                kwargs['block_config'] = [3]*4
            else:
                assert False

            self.classifier = models.DenseNet(
                **kwargs
            )
        else:
            import ipdb; ipdb.set_trace()

        inner_opt_class_name = self.args.inner_optim['class']
        inner_opt_class = hydra.utils.get_class(inner_opt_class_name)
        kwargs = dict(self.args.inner_optim.params)
        if 'Adam' in inner_opt_class_name:
            kwargs['betas'] = (kwargs['beta1'], kwargs['beta2'])
            del kwargs['beta1'], kwargs['beta2']
        if args.learnable_inner_opt_params:
            param_groups = [
                {'params': p, 'lr': kwargs['lr']} for p in self.classifier.parameters()
            ]
            self.inner_opt = inner_opt_class(param_groups, **kwargs)
            t = higher.optim.get_trainable_opt_params(self.inner_opt)
            self.lrs = nn.ParameterList(map(
                nn.Parameter,
                t['lr']
            ))
            if 'Adam' in inner_opt_class_name:
                self.betas = nn.ParameterList(list(map(
                    nn.Parameter,
                    [item for sublist in t['betas'] for item in sublist]
                )))
        else:
            params = self.classifier.parameters()
            self.inner_opt = inner_opt_class(params, **kwargs)

        # self.use_cuda = args.use_cuda
        self.device = device
        self.args = args
        self.to(device)
        print("Outer Loop parameters")
        param_shapes = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)
                param_shapes.append(param.shape)
        print(f'n_params: {sum(map(np.prod, param_shapes))}')

        self.optimizer = optim.Adam(
            self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer, T_max=self.args.total_epochs,
            eta_min=self.args.min_learning_rate
        )

    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.args.number_of_training_steps_per_iter)) * (
                1.0 / self.args.number_of_training_steps_per_iter)
        decay_rate = 1.0 / self.args.number_of_training_steps_per_iter / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.args.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.args.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if self.args.enable_inner_loop_optimizable_bn_params:
                    param_dict[name] = param.to(device=self.device)
                else:
                    if "norm_layer" not in name:
                        param_dict[name] = param.to(device=self.device)

        return param_dict

    def get_across_task_loss_metrics(self, total_losses, total_accuracies):
        losses = dict()

        losses['loss'] = torch.mean(torch.stack(total_losses))
        losses['accuracy'] = np.mean(total_accuracies)

        return losses

    def forward(self, data_batch, epoch, use_second_order,
                use_multi_step_loss_optimization, num_steps, training_phase):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        [b, ncs, spc] = y_support_set.shape

        self.num_classes_per_set = ncs

        total_losses = []
        total_accuracies = []
        per_task_target_preds = [[] for i in range(len(x_target_set))]
        self.classifier.zero_grad()
        for task_id, (x_support_set_task, y_support_set_task,
                      x_target_set_task, y_target_set_task) in \
                enumerate(zip(x_support_set, y_support_set,
                              x_target_set, y_target_set)):
            task_losses = []
            task_accuracies = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()

            n, s, c, h, w = x_target_set_task.shape
            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            y_support_set_task = y_support_set_task.view(-1)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1)

            with higher.innerloop_ctx(
                self.classifier, self.inner_opt, copy_initial_weights=False,
                track_higher_grads=training_phase,
            ) as (fnet, diffopt):
                for p in self.classifier.parameters():
                    self.inner_opt.state[p] = copy.deepcopy(self.optimizer.state[p])
                for num_step in range(num_steps):
                    support_preds = fnet(x_support_set_task)
                    support_loss = F.cross_entropy(
                        input=support_preds, target=y_support_set_task)

                    if self.args.learnable_inner_opt_params:
                        if 'Adam' in self.args.inner_optim['class']:
                            betas = [self.betas[i:i+2] for i in range(0, len(self.betas), 2)]
                            diffopt.step(
                                support_loss, override={'lr': self.lrs, 'betas': betas}
                            )
                        else:
                            diffopt.step(
                                support_loss, override={'lr': self.lrs},
                            )
                    else:
                        diffopt.step(support_loss)

                    if use_multi_step_loss_optimization and training_phase and \
                            epoch < self.args.multi_step_loss_num_epochs:
                        target_preds = fnet(x_target_set_task)
                        target_loss = F.cross_entropy(
                            input=target_preds, target=y_target_set_task)
                        task_losses.append(
                            per_step_loss_importance_vectors[num_step] * target_loss)
                    else:
                        if num_step == (self.args.number_of_training_steps_per_iter - 1):
                            target_preds = fnet(x_target_set_task)
                            target_loss = F.cross_entropy(
                                input=target_preds, target=y_target_set_task)
                            task_losses.append(target_loss)

            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            _, predicted = torch.max(target_preds.data, 1)

            accuracy = predicted.float().eq(y_target_set_task.data.float()).cpu().float()
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            total_accuracies.extend(accuracy)

        losses = self.get_across_task_loss_metrics(
            total_losses=total_losses,
            total_accuracies=total_accuracies
        )

        for idx, item in enumerate(per_step_loss_importance_vectors):
            losses['loss_importance_vector_{}'.format(idx)] = item.detach().cpu().numpy()

        return losses, per_task_target_preds

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def train_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(
            data_batch=data_batch, epoch=epoch,
            use_second_order=self.args.second_order and
            epoch > self.args.first_order_to_second_order_epoch,
            use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
            num_steps=self.args.number_of_training_steps_per_iter,
            training_phase=True)
        return losses, per_task_target_preds

    def evaluation_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(
            data_batch=data_batch, epoch=epoch, use_second_order=False,
            use_multi_step_loss_optimization=True,
            num_steps=self.args.number_of_evaluation_steps_per_iter,
            training_phase=False)

        return losses, per_task_target_preds

    def meta_update(self, loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        if 'imagenet' in self.args.dataset.name:
            for name, param in self.classifier.named_parameters():
                if param.requires_grad:
                    param.grad.data.clamp_(-10, 10)  # not sure if this is necessary, more experiments are needed
        self.optimizer.step()

        if self.args.learnable_inner_opt_params:
            for lr in self.lrs:
                lr.data[lr < 1e-4] = 1e-4
            if 'Adam' in self.args.inner_optim['class']:
                for beta in self.betas:
                    beta.data[beta < 1e-4] = 1e-4
                    beta.data[beta > 0.99] = 0.99


    def run_train_iter(self, data_batch, epoch):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        epoch = int(epoch)
        self.scheduler.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        self.train()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_target_preds = self.train_forward_prop(
            data_batch=data_batch, epoch=epoch)

        self.meta_update(loss=losses['loss'])
        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.optimizer.zero_grad()
        self.zero_grad()
        losses['loss'] = losses['loss'].detach().cpu()

        return losses, per_task_target_preds

    def write_inner_opt_stats(self):
        if self.args.learnable_inner_opt_params:
            lrs = torch.stack(list(self.lrs)).cpu().detach().numpy()
            f = open('lrs.csv', 'a')
            f.write(','.join(map(str, lrs)) + '\n')
            f.close()
            if 'Adam' in self.args.inner_optim['class']:
                betas = torch.stack(list(self.betas)).cpu().detach().numpy()
                f = open('betas.csv', 'a')
                f.write(','.join(map(str, betas)) + '\n')
                f.close()

    def run_validation_iter(self, data_batch):
        """
        Runs an outer loop evaluation step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """

        # if self.training:
        #     self.eval()
        self.train()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_target_preds = self.evaluation_forward_prop(
            data_batch=data_batch, epoch=self.current_epoch)

        # losses['loss'].backward() # uncomment if you get the weird memory error
        # self.zero_grad()
        # self.optimizer.zero_grad()
        losses['loss'] = losses['loss'].detach().cpu()

        return losses, per_task_target_preds

    def save_model(self, model_save_dir, state):
        """
        Save the network parameter state and experiment state dictionary.
        :param model_save_dir: The directory to store the state at.
        :param state: The state containing the experiment state and the network. It's in the form of a dictionary
        object.
        """
        state['network'] = self.state_dict()
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):
        """
        Load checkpoint and return the state dictionary containing the network state params and experiment state.
        :param model_save_dir: The directory from which to load the files.
        :param model_name: The model_name to be loaded from the direcotry.
        :param model_idx: The index of the model (i.e. epoch number or 'latest' for the latest saved model of the current
        experiment)
        :return: A dictionary containing the experiment state and the saved model parameters.
        """
        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        state = torch.load(filepath)
        state_dict_loaded = state['network']
        self.load_state_dict(state_dict=state_dict_loaded)
        return state
