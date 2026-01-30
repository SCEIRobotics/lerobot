#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Dict, Any, List

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset, make_val_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, ACTION, MAX_ACTION_DIM
import torch.nn.functional as F
import gc


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        if isinstance(optimizer, dict):
            for opt in optimizer.values():
                opt.step()
        else:
            optimizer.step()

    if isinstance(optimizer, dict):
        for opt in optimizer.values():
            opt.zero_grad(set_to_none=True)
    else:
        optimizer.zero_grad(set_to_none=True)

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        if isinstance(lr_scheduler, dict):
            for sched in lr_scheduler.values():
                sched.step()
        else:
            lr_scheduler.step() 
    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer['dit'].param_groups[0]["lr"] if isinstance(optimizer, dict) else optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    # if True:
    #     repo_ids = []
    #     roots = []
    #     project_name = 'interna1_merge_all' #'sim_lerobot' # lift2/franka interna1_merge_all
    #     project_dir = f"{cfg.dataset.root}/{project_name}"
    #     repo_names = os.listdir(project_dir)
    #     for repo_name in repo_names:
    #         if repo_name == 'interna1_lift2_processed_diff':
    #             continue
    #         repo_ids.append(f'{project_name}/{repo_name}')
    #         roots.append(f'{project_dir}/{repo_name}')
    #     cfg.dataset.repo_id = repo_ids
    #     cfg.dataset.root = roots
    # print(cfg.dataset.repo_id)
    # print(cfg.dataset.root)
    if True:
        repo_ids = []
        roots = []
        project_name = 'datasets' #'sim_lerobot' # lift2/franka interna1_merge_all
        project_dir = f"{cfg.dataset.root}/{project_name}"
        repo_names = os.listdir(project_dir)
        for repo_name in repo_names:
            if repo_name != 'aloha_sim_transfer_cube_scripted_train' and repo_name != 'aloha_sim_transfer_cube_scripted_val':
                continue
            repo_ids.append(f'{project_name}/{repo_name}')
            roots.append(f'{project_dir}/{repo_name}')
        cfg.dataset.repo_id = repo_ids
        cfg.dataset.root = roots
        cfg.dataset.root_val = roots
    
    print(cfg.dataset.repo_id)
    print(cfg.dataset.root)
    print(cfg.dataset.root_val)

    cfg.validate()
    # import pdb; pdb.set_trace()
    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs
        from accelerate import DataLoaderConfiguration
        dataloader_config = DataLoaderConfiguration(
            # split_batches=True,  
            use_seedable_sampler=False 
            )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False, 
            # dataloader_config=dataloader_config,
            gradient_accumulation_steps=4,
            kwargs_handlers=[ddp_kwargs])

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    cfg.policy.device = str(device)
    cfg.policy.mixed_precision = str(accelerator.mixed_precision)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        datasets = make_dataset(cfg)
        val_datasets = make_val_dataset(cfg) 

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        datasets = make_dataset(cfg)
        val_datasets = make_val_dataset(cfg) 
    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    if is_main_process:
        logging.info("Creating policy")
    
    ds_metas = [ds.meta for ds in datasets]
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=ds_metas,
        rename_map=cfg.rename_map,
        )

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env)
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{sum([dataset.num_frames for dataset in datasets])=} ({format_big_number(sum([dataset.num_frames for dataset in datasets]))})")
        logging.info(f"{sum([dataset.num_episodes for dataset in datasets])=}")
        num_processes = accelerator.num_processes
        gradient_accumulation_steps = accelerator.gradient_accumulation_steps
        effective_bs = cfg.batch_size * num_processes * gradient_accumulation_steps
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} x {gradient_accumulation_steps} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    shuffle = True
    sampler = None
    if not cfg.dataset.streaming:
        raise NotImplementedError("Multi-dataset training is just supported for streaming.")
    
    # from lerobot.datasets.streaming_dataset import MixedIterableDataset 
    # from lerobot.datasets.utils import FlowerDataCollator
    # from torch.utils.data import ChainDataset
    # ratios = cfg.policy.dataset_ratios or [1.0] * len(datasets)
    # mix_dataset = MixedIterableDataset(datasets, ratios)
    # mix_dataset = ChainDataset(datasets)
    # dataloader = torch.utils.data.DataLoader(
    #     mix_dataset,
    #     num_workers=cfg.num_workers,
    #     batch_size=cfg.batch_size,
    #     shuffle=shuffle and not cfg.dataset.streaming,
    #     sampler=sampler,
    #     collate_fn=FlowerDataCollator(),
    #     pin_memory=device.type == "cuda",
    #     drop_last=True,
    #     prefetch_factor=2 if cfg.num_workers > 0 else None,
    #     )
    from lerobot.datasets.utils import FlowerDataCollator
    dataloaders_ = []
    sample_weights = []
    dataset_sizes = []
    # sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
    # sample_weights = np.array(sample_weights) / np.sum(sample_weights)
    import numpy as np
    for sub_idx in range(len(datasets)):
        dataloader = torch.utils.data.DataLoader(
            datasets[sub_idx],
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=shuffle and not cfg.dataset.streaming,
            sampler=sampler,
            collate_fn=FlowerDataCollator(),
            pin_memory=device.type == "cuda",
            drop_last=True,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
            )
        dataloaders_.append(dataloader)
        sample_weights.append(datasets[sub_idx].weight)
        dataset_sizes.append(datasets[sub_idx].meta.total_frames)
    sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
    sample_weights = np.array(sample_weights) / np.sum(sample_weights)
    # import pdb; pdb.set_trace()

    from torch.utils.data import ConcatDataset
    val_mix_dataset = ConcatDataset(val_datasets)
    val_dataloader = torch.utils.data.DataLoader(
        val_mix_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=None,
        collate_fn=FlowerDataCollator(),
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    dataloaders = []
    for dataloader in dataloaders_:
        dataloader = accelerator.prepare(dataloader)
        dataloaders.append(dataloader)
    policy, optimizer, lr_scheduler, val_dataloader = accelerator.prepare(
        policy, optimizer, lr_scheduler, val_dataloader
    )
    dl_iters = [cycle(dl) for dl in dataloaders]
    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        sum([dataset.num_frames for dataset in datasets]),
        sum([dataset.num_episodes for dataset in datasets]),
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info("Start offline training on a fixed dataset")

    for batch_idx in range(step, cfg.steps):
        with accelerator.accumulate(policy):
            start_time = time.perf_counter()
            st = time.time()
            import random
            dataloader_idx = random.choices(range(len(dataloaders)), weights=sample_weights)[0]
            batch = next(dl_iters[dataloader_idx])
            batch = {
                k: v.to(cfg.policy.device, non_blocking=True) 
                if isinstance(v, torch.Tensor)
                else v 
                for k, v in batch.items()
            }
            train_tracker.dataloading_s = time.perf_counter() - start_time
            pt = time.time()
            train_tracker, output_dict = update_policy(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                accelerator=accelerator,
                lr_scheduler=lr_scheduler,
            )
            ut = time.time()
            print(f"use dataloader_idx: {dataloader_idx}, dataloading_s: {pt-st:.3f}, update_s: {ut - pt:.3f}")
        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        # flower pret没有选择在同步梯度的时候更新step
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
        is_valid_step = cfg.valid_freq > 0 and step % cfg.valid_freq == 0

        if is_valid_step:
            policy.eval()
            val_loss = 0
            for vla_idx, val_batch in enumerate(val_dataloader):
                if vla_idx >= 10:
                    break
                loss = policy.validate_action(val_batch)
                val_loss += loss.item()
            print(f"val_loss: {loss.item():.3f}")
            # val_loss /= len(val_dataloader)

            val_loss /= (val_dataloader+1)
            print(val_loss)
            logging.info(f"val_loss: {val_loss:.3f}")
            policy.train()

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()
            torch.cuda.empty_cache()
            gc.collect()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    train()


if __name__ == "__main__":
    import setproctitle
    setproctitle.setproctitle("lerobot")
    main()
