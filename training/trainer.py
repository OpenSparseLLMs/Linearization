import sys
import os
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
import os
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments

from peft import PeftModel

class DefaultTrainer():
    # code is modified from: https://github.com/HazyResearch/lolcats/blob/main/src/trainer/default_lm.py
    def __init__(self, model, train_loader, eval_loader, args, optimizers, tokenizer, config):
        super().__init__()
        self.model = model
        self.args = args
        self.tokenizer = tokenizer
        self.config = config
        self.type = 'default'

        self.step = 0  # Total steps taken
        self.grad_step = 0  # Total gradient updates
        self.compute_loss_backprop = False  # Whether we backprop in self.compute_loss

        self.optimizer, self.scheduler = optimizers
        self.scheduler_step_after_epoch = True
        # Dataloaders
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.device = model.device
        wandb = None
        self.wandb = wandb

        # args
        self.metric_for_best_model = self.args.metric_for_best_model
        self.num_train_epochs = self.args.num_train_epochs
        self.gradient_accumulation_steps = self.args.gradient_accumulation_steps
        self.eval_strategy = self.args.eval_strategy
        self.greater_is_better = self.args.greater_is_better
        self.is_better = (lambda x, y: x > y if self.args.greater_is_better else x < y)
        self.load_best_model_at_end = self.args.load_best_model_at_end
        self.logging_steps = self.args.logging_steps
        self.max_steps = self.args.max_steps
        self.eval_steps = self.args.eval_steps

        max_eval_batches = -1
        print_samples = False
        initial_eval = True
        self.max_eval_batches = max_eval_batches
        self.print_samples = print_samples
        self.initial_eval = initial_eval
        self.save_total_limit = self.args.save_total_limit 
        self.save_steps = self.args.save_steps # num_save_ckpt_steps

        # Saving metrics
        self.train_metrics = {'train/loss': None, 
                              'train/epoch': None, 
                              'train/step': None}
        self.eval_metrics = {self.metric_for_best_model: None}
        self.eval_metrics_by_step = {'eval_step': []}  # save all eval metrics
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
            
        save_results = True
        save_checkpoints = True
        
        self.save_results = save_results
        self.results_path = None
        self.best_val_metric = 0 if self.greater_is_better else 1e10
        self.best_val_metric_epoch = 0
        self.best_val_metric_step = 0
        if save_checkpoints:  # Also initializes best_val_metrics
            self.init_checkpointing(config=config)

    def train(self) -> nn.Module:
        """
        Entire training run
        """
        model = self.model
        pbar = tqdm(range(self.num_train_epochs), leave=False, colour='white', desc='Training')
        for ix, epoch in enumerate(pbar):
            model, early_stopping = self.train_step(model, epoch)
            if self.eval_strategy == 'epoch':
                _eval_metrics = self.eval_step(model, step=self.grad_step)
                print(f'Epoch {ix} metrics:', _eval_metrics)
            if early_stopping:
                break
                
        if self.load_best_model_at_end:  # Return best checkpoint
            try:
                model.from_pretrained(self.best_val_checkpoint_path)
                print(f'-> Loading best checkpoint from {self.best_val_checkpoint_path}')
            except FileNotFoundError as e:
                print(e)
                print('-> Returning most recent model instead')
        return model            
    
    def train_step(self, model, epoch) -> nn.Module:
        if self.gradient_accumulation_steps is None:
            accum_iter = 1
        else:
            accum_iter = self.gradient_accumulation_steps

        model.train()
        model.zero_grad()        
        pbar = tqdm(self.train_loader, leave=False, colour='blue', desc=f'-> Training (epoch {epoch} / {self.args.num_train_epochs})')
        total_loss = 0
        eval_for_step = False

        # Initial eval
        if self.initial_eval:
            print('')
            print('-> Initial eval')
            self.compute_eval_metrics(model, step=self.grad_step)
        
        # model.to(self.device)
        for ix, data in enumerate(pbar):
            loss, train_metrics = self.compute_loss(model, data, return_outputs=True)
            loss /= accum_iter
            if not self.compute_loss_backprop:
                # loss.backward() did not occur in compute_loss
                try:
                    with torch.autograd.set_detect_anomaly(True):
                        loss.backward()
                except Exception as e:
                    breakpoint()
            if (self.step + 1) % accum_iter == 0:  # and self.step != 0:
                self.optimizer.step()
                if not self.scheduler_step_after_epoch and self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.grad_step += 1
                if not self.compute_loss_backprop:
                    loss = loss.detach().cpu().item()
            
            self.step += 1
            if not isinstance(loss, float):
                total_loss += loss.item()
            else:
                total_loss += loss
            desc = f"Training epoch {epoch} | loss: {total_loss / (ix + 1):.3f} | lr: {self.optimizer.param_groups[0]['lr']:.5f}"
            desc += f' | gradient step: {self.grad_step}'
            for k, v in train_metrics.items():
                desc += f' | {k}: {v:.3f}'
            pbar.set_description(desc)

            # Logging
            if (self.grad_step) % (self.logging_steps):
                self.train_metrics['train/loss'] = loss.item() if not isinstance(loss, float) else loss
                self.train_metrics['train/epoch'] = epoch
                self.train_metrics['train/step'] = self.grad_step
                self.train_metrics['train/lr'] = self.optimizer.param_groups[0]['lr']
                for k, v in train_metrics.items():
                    self.train_metrics[f'train/{k}'] = v
                
                if self.wandb is not None:
                    self.wandb.log(self.train_metrics, step=self.grad_step)

            if self.eval_strategy == 'steps':
                if (self.grad_step % self.eval_steps == 0 and self.grad_step > 0 and not eval_for_step):
                    _eval_metrics = self.eval_step(model, step=self.grad_step)
                    print(f'Grad Step {self.grad_step} eval metrics:', _eval_metrics)
                    eval_for_step = True
                    model.train()  # Need to set back to train mode
                elif self.grad_step == 0 and self.save_steps < 1000 and not eval_for_step:  # hack for micros
                    _eval_metrics = self.eval_step(model, step=self.grad_step)
                    print(f'Grad Step {self.grad_step} eval metrics:', _eval_metrics)
                    eval_for_step = True
                    model.train()  # Need to set back to train mode
                    
                elif self.grad_step % self.eval_steps == 0 and self.grad_step > 0 and eval_for_step:
                    pass
                else:
                    if self.grad_step > 0:
                        eval_for_step = False
            if self.grad_step == self.max_steps:
                early_stopping = True
                return model, early_stopping
        
        early_stopping = False
        return model, early_stopping

    
    def eval_step(self, model: nn.Module, step: int = None, **kwargs: any) -> dict[any]:
        """
        Evaluation loop over one epoch
        """
        with torch.no_grad():
            self.eval_metrics = self.compute_eval_metrics(model, step=step, **kwargs)
            val_metric = self.eval_metrics[self.metric_for_best_model]

            # Save results
            if self.wandb is not None:  # log to WandB
                self.wandb.log(self.eval_metrics, step=self.grad_step)

            if self.results_path is not None:  # log to local file
                self.eval_metrics_by_step['eval_step'].append(step)
                for k, v in self.eval_metrics.items():
                    if k not in self.eval_metrics_by_step:
                        self.eval_metrics_by_step[k] = [v]
                    else:
                        self.eval_metrics_by_step[k].append(v)
                # Inefficient, but log for experiments results
                pd.DataFrame(self.eval_metrics_by_step).to_csv(self.results_path)

            # Save best metric and checkpoint
            if self.grad_step % self.eval_steps == 0 and step > 0:
                if self.is_better(val_metric, self.best_val_metric):
                    self.best_val_metric = val_metric
                    self.best_val_metric_step = self.grad_step

                    save_path = self.save_path + '/iter' + '_' + str(step)
                    self.best_val_checkpoint_path = save_path
                    model.save_pretrained(save_path)
                    self.tokenizer.save_pretrained(save_path)
                    print(f'\n-> Saved best model checkpoint to: {save_path}!')

            if self.grad_step % self.save_steps == 0 and step > 0:

                save_path = self.save_path + '/' + self.type + '_' + str(step)
                self.best_val_checkpoint_path = save_path
                model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                print(f'\n-> Saved model checkpoint to: {save_path}!')
            
            if self.scheduler_step_after_epoch and self.scheduler is not None:
                self.scheduler.step(val_metric)
            return self.eval_metrics

    def compute_eval_metrics(self, 
                            model: nn.Module, step: int,
                            max_batches: int = None,
                            dataloader: DataLoader = None,
                            **kwargs: any) -> dict[any]:
        """
        One evaluation loop over a validation dataset
        """
        max_batches = (self.max_eval_batches if max_batches is None else max_batches)
        dataloader = self.eval_loader if dataloader is None else dataloader
        pbar = tqdm(dataloader, leave=False, colour='green', desc=f'Evaluating at step {step}')

        model.eval()
        step_loss = 0
        step_eval_metrics = {}
        with torch.no_grad():
            for ix, data in enumerate(pbar):
                loss, eval_metrics = self.compute_loss(model, data, return_outputs=True)
                if not self.compute_loss_backprop:
                    loss = loss.item()  # otherwise already float
                if ix == 0:
                    step_eval_metrics[self.metric_for_best_model] = [loss]
                    for k, v in eval_metrics.items():
                        step_eval_metrics[f'eval/{k}'] = [v]
                else:
                    step_eval_metrics[self.metric_for_best_model].append(loss)
                    for k, v in eval_metrics.items():
                        step_eval_metrics[f'eval/{k}'].append(v)
                        
                step_loss += loss
                desc = f"Evaluating at step {step} | loss: {step_loss / (ix + 1):.3f}"
                if self.optimizer is not None:
                    desc += f" | lr: {self.optimizer.param_groups[0]['lr']:.5f}"
                pbar.set_description(desc)
                if ix == max_batches:
                    break

            # Average over batches
            for k, v in step_eval_metrics.items():
                step_eval_metrics[k] = sum(v) / len(v)
            print(f'Eval step {step}:', step_eval_metrics)
            del loss
            torch.cuda.empty_cache()
        return step_eval_metrics

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = {k: v.to(model.device) for k, v in inputs.items() if k != 'labels'}
        outputs = model(**inputs, output_attentions=True)
        
        outputs = outputs.attentions # tuple [num_decoder_layers, 2, B, H, L, L]
        loss_mse = 0
        self.mse_factor = 1000
        self.criterion_mse = nn.MSELoss(reduction='mean')
        n_layers = 0  # Number of layers to distill

        for layer_idx, attns in enumerate(outputs):
            if attns is not None:
                loss_mse += self.criterion_mse(attns[0], attns[1])
                n_layers += 1

        if n_layers > 0:
            loss_mse = loss_mse / n_layers * self.mse_factor
        loss = loss_mse
        outputs = {'loss_mse': loss_mse.item() if self.mse_factor > 0 else 0, 
                   'mse_factor': self.mse_factor}
        
        return (loss, outputs) if return_outputs else loss

    def init_checkpointing(self, config) -> None:
        self.save_path = config.train.output_dir
        self.best_val_checkpoint_path = config.train.output_dir

        # Best metric setup
        self.best_val_metric = 0 if self.greater_is_better else 1e10
        self.best_val_metric_epoch = 0
        self.best_val_metric_step = 0
        self.best_train_metric = 0 if self.greater_is_better else 1e10
        self.best_train_metric_epoch = 0
        self.best_train_metric_step = 0
        self.metric_for_best_model = self.metric_for_best_model
        if self.metric_for_best_model is not None:
            if 'eval' not in self.metric_for_best_model:
                self.metric_for_best_model = f'eval/{self.metric_for_best_model}'

class FinetuneTrainer(DefaultTrainer):
    def __init__(self, model, train_loader, eval_loader, args, optimizers, tokenizer, config):
        super().__init__(model, train_loader, eval_loader, args, optimizers, tokenizer, config)
        self.type = 'finetune'

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_keys = {'input_ids', 'attention_mask'}
        data = {k: v.to(model.device) for k, v in inputs.items() if k in input_keys}  
        outputs = model(**data, output_attentions=False)
        outputs = outputs.get('logits')[..., :-1, :].contiguous()
        targets = inputs.get('labels')[..., 1:].contiguous()
        # Flatten and compute cross-entropy loss
        outputs = outputs.view(-1, outputs.shape[-1])
        targets = targets.view(-1).to(outputs.device)
        loss = self.criterion(outputs, targets)
        
        targets = targets.cpu()
        outputs = outputs.cpu()
        outputs = {'ppl': torch.exp(loss).item(), 'seq_len': targets.shape[-1] + 1}
        return (loss, outputs) if return_outputs else loss
    