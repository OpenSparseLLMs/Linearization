import sys
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import fla
import liger
import lolcats
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import TrainingArguments
from peft import LoraConfig, TaskType, PeftModel, get_peft_model

from training.trainer import DefaultTrainer, FinetuneTrainer
from training.utils import get_optimizer_and_scheduler, count_model_params
from training.dataloader import load_data


def train(config):

    trainer = FinetuneTrainer
    if config.model.name == "liger_gla":
        from liger.models.liger_gla import LigerGLAConfig
        liger_model_config = LigerGLAConfig()
    elif config.model.name == "liger_gsa":
        from liger.models.liger_gsa import LigerGSAConfig
        liger_model_config = LigerGSAConfig()
    elif config.model.name == "liger_qwen25_gla":
        from liger.models.liger_qwen2_gla import LigerQwen2GLAConfig
        liger_model_config = LigerQwen2GLAConfig()
    elif config.model.name == "liger_qwen3_gla":
        from liger.models.liger_qwen3_gla import LigerQwen3GLAConfig
        liger_model_config = LigerQwen3GLAConfig()
    elif config.model.name == "liger_qwen3_moe_gla":
        from liger.models.liger_qwen3_moe_gla import LigerQwen3MoeGLAConfig
        liger_model_config = LigerQwen3MoeGLAConfig()
    elif config.model.name == "lolcats_at":
        # first stage: attention transfer
        from lolcats.models.lolcats import LolcatsConfig
        liger_model_config = LolcatsConfig()
        trainer = DefaultTrainer
    elif config.model.name == "lolcats_ar":
        # second stage
        from lolcats.models.lolcats import LolcatsConfig
        liger_model_config = LolcatsConfig()
    else:
        raise NotImplementedError(config.model.name)
    
    model_config = liger_model_config
    model = AutoModelForCausalLM.from_pretrained(
        config.model.pretrained_model_name_or_path, 
        config=model_config, 
        device_map="cuda"
    ).to(torch.bfloat16)


    print("Model config:")
    print(model_config)
    print("Model:")
    print(model)

    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # Allow batched inference

    for name, param in model.named_parameters():
        param.requires_grad = False
        if "train_qk" in config.train and config.train.train_qk:
            if "self_attn.q_proj" in name:
                param.requires_grad = True
            elif "self_attn.k_proj" in name:
                param.requires_grad = True
        if "train_v" in config.train and config.train.train_v and "self_attn.v_proj" in name:
            param.requires_grad = True
        if "train_o" in config.train and config.train.train_o and "self_attn.o_proj" in name:
            param.requires_grad = True

    # LoRA finetune
    target_modules = []
    if  "train_qk" in config.train and config.train.train_qk and config.train.train_qk_lora:
        target_modules.append("self_attn.q_proj")
        target_modules.append("self_attn.k_proj")
    if  "train_v" in config.train and config.train.train_v and config.train.train_v_lora:
        target_modules.append("self_attn.v_proj")
    if  "train_o" in config.train and config.train.train_o and config.train.train_o_lora:
        target_modules.append("self_attn.o_proj")
    # lolcats attention transfer
    if config.model.name == "lolcats_at":
        for name, param in model.named_parameters():
            if "feature_map" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    
    if len(target_modules) != 0:
        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, target_modules=target_modules)
        model = get_peft_model(model, peft_config=lora_config)

    # print trainable params count
    trainable_params = count_model_params(model, requires_grad=True)
    total_params = count_model_params(model, requires_grad=False)
    print(f"Model trainable params: {trainable_params}")
    print(f"Model total params: {total_params}")
    print(f"trainable%: {trainable_params / total_params}")

    gradient_accumulation_steps = config.data.batch_size // config.data.micro_batch_size

    print("Preparing data...")

    dataloaders  = load_data(config)
    train_loader = dataloaders["train"]
    eval_loader  = dataloaders["validation"]

    print("Building trainer...")    

    training_args = TrainingArguments(
            per_device_train_batch_size=config.data.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=config.train.epochs,
            learning_rate=config.train.lr,
            bf16=True,
            max_grad_norm=config.train.max_grad_norm,
            logging_steps=1,
            optim=config.train.optim,
            eval_strategy="steps" if config.data.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if config.data.val_set_size > 0 else None,
            save_steps=1000,
            logging_dir=config.train.output_dir,
            output_dir=config.train.output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if config.data.val_set_size > 0 else False,
            # default trainer args
            greater_is_better = False,
            metric_for_best_model = 'eval/loss',
            # wandb
            report_to="none" # wandb off "wandb"
        )
    
    trainer = trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        args=training_args,
        optimizers=get_optimizer_and_scheduler(model, config),
        tokenizer=tokenizer,
        config=config
    )

    print("Train start")
    best_model = trainer.train()
    save_path = trainer.save_path + '/best'
    best_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f'\n-> Saved best model checkpoint to: {save_path}!')

    print("Train over")
