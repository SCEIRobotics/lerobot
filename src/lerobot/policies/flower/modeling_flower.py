#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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
"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

TODO(alexander-soare):
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
"""

import math
from collections import deque
from collections.abc import Callable

import einops
import functools
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from typing import Any, Dict, Optional, Tuple, Collection, List
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from timm.layers.mlp import Mlp
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.policies.flower.configuration_flower import FlowerConfig
from lerobot.policies.flower.utils import generate_policy_prompt, ActionIndex
from lerobot.policies.flower.transformers_flower import (
    TimestepEmbedder,
    SharedAdaLNController,
    RmsNorm,
    FreqEmbedder,
    ActionSpaceEmbedderParameter,
    ZeroEncoder,
    FlowBlock, 
    stateless_norm
)
from torchvision.utils import save_image
DEFAULT_DTYPE = torch.bfloat16
# torch.bfloat16
# torch.float32

class FlowerPolicy(PreTrainedPolicy):
    """
    Flower Policy as per "FLOWER: Democratizing Generalist Robot Policies with Efficient Vision-Language-Action Flow Policies"
    (paper: https://arxiv.org/pdf/2509.04996, code: https://intuitive-robots.github.io/flower_vla/).
    """

    config_class = FlowerConfig
    name = "flower" 

    def __init__(
        self,
        config: FlowerConfig,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.flower = FlowerModel(config)
        self.reset()

    def get_optim_params(self) -> dict:
        # return self.flower.parameters()
        """Get parameter groups for optimizer"""
        no_decay = ['bias', 'LayerNorm', 'layernorm', 'ln', 'norm']
        decay_group = []
        no_decay_group = []

        # Collect all parameters, excluding VLM if frozen
        for name, param in self.flower.named_parameters():
            if param.requires_grad:
                if any(nd in name.lower() for nd in no_decay):
                    no_decay_group.append(param)
                else:
                    decay_group.append(param)

        return [
            {"params": decay_group, "weight_decay": self.config.optimizer_weight_decay},
            {"params": no_decay_group, "weight_decay": 0.0}
        ]

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
            'task': deque(maxlen=self.config.n_obs_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def validate_action_chunk_(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        # 改为不使用队列，直接使用batch
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # import pdb; pdb.set_trace()
        actions = self.flower.generate_actions(batch, noise=noise)

        return actions
    
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        # stack n latest observations from the queue
        # import pdb; pdb.set_trace()
        # batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        # for k in batch:
        #     if k in self._queues:
        #         if k=='task':
        #             # batch[k]=['Push the T-shaped block onto the T-shaped target.' for i in range(500)]
        #             batch[k]=['Pick up the cube with the right arm and transfer it to the left arm.' for i in range(16)]
        #         else:
        #             batch[k] = torch.stack(list(self._queues[k]), dim=1)
        batch['observation.state']=batch['observation.state'].unsqueeze(0)
        batch['observation.images']=batch['observation.images.image'].unsqueeze(0).unsqueeze(0)
        batch_size, n_obs_steps, cam, channels, height, width = batch['observation.images'].shape
        x_reshaped = batch['observation.images'].view(
                    batch_size*n_obs_steps*cam,
                    channels,
                    height,
                    width,
                )
        x_reshaped = F.interpolate(x_reshaped, size=(224,224), mode='bilinear')
        batch['observation.images'] = x_reshaped.view(
                    batch_size,
                    n_obs_steps,
                    cam,
                    channels,
                    224,
                    224,
                )
        # save_image(batch['observation.images'][0,0,0,:,:,:], 'test.png')

        actions = self.flower.generate_actions(batch, noise=noise)

        return actions
    
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # NOTE: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
        # 处理image features
        image_keys = []
        VALID_PREFIXES = ('observation.image', 'images')
        EXCLUDE_KEYWORDS = ('_is_pad',)
        for key in batch:
            if not any(kw in key for kw in EXCLUDE_KEYWORDS) and key.startswith(VALID_PREFIXES):
                if key!='observation.images.rgb.head':
                    continue
                image_keys.append(key)
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in image_keys], dim=-4)
        loss = self.flower.compute_loss(batch)
        print(f"loss: {loss}")
        # no output_dict so returning None
        return loss, None


class FlowerModel(nn.Module):
    def __init__(self, config: FlowerConfig):
        super().__init__()
        # super().__init__(*args, **kwargs)
        self.config = config
        self.num_inference_steps = config.num_inference_steps
        self.device = config.device
        
        # Initialize configurations: 已移到configuration_flower
        # Set task prompt format: 已移到stream dataset 处理
        
        # Setup VLM and core components
        self._setup_vlm(
            config.vlm_path, 
            config.freeze_vision_tower, 
            config.freeze_florence
            )
        self.config.hidden_dim = self.vlm.config.text_config.d_model
        self.vlm_latent_dim = self.config.hidden_dim

        # Setup DiT components
        self.action_space_index = ActionIndex()
        self._setup_dit_components()
        
        # Load pretrained weights if specified，只用于加载flower原始训练框架预训练模型
        if config.load_pretrained and config.pretrained_model_path is not None:
            self._load_pretrained_weights(config.pretrained_model_path)

    # ========= init  ============
    def _setup_vlm(self, vlm_path: str, freeze_vision_tower: bool, freeze_florence: bool):
        """Initialize and configure the Florence-2 VLM"""
        print(f"Loading Florence-2 from {vlm_path}")
        
        self.vlm = AutoModelForCausalLM.from_pretrained(vlm_path, trust_remote_code=True)
        
        # Handle parameter freezing
        if freeze_florence:
            for param in self.vlm.parameters():
                param.requires_grad = False
        elif not freeze_vision_tower:
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = True

        # Setup processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        
        # Create prompt embedding
        self.prompt_embeds = self._create_prompt_embed("<Flow>")
        
        # Remove unnecessary components
        del self.vlm.language_model.model.decoder
        del self.vlm.language_model.lm_head
        
        # Setup token dropout
        self.vlm_token_dropout = nn.Dropout(self.config.token_dropout)

    def _setup_dit_components(self):
        """Setup DiT model components"""

        self.action_encoders = nn.ModuleDict()
        self.action_decoders = nn.ModuleDict()
        if self.config.use_proprio:
            self.proprio_encoders = nn.ModuleDict()
            
        self.adaln = nn.ModuleDict() if self.config.action_type_adaln else None

        # Core components
        self.cond_linear = nn.Linear(self.config.hidden_dim, self.config.dit_dim, bias=False)
        self.t_embedder = TimestepEmbedder(self.config.dit_dim)
        self.cond_norm = RmsNorm(self.config.hidden_dim)
        self.frequency_embedder = FreqEmbedder(self.config.dit_dim)
        self.action_space_embedder = ActionSpaceEmbedderParameter(
            self.config.dit_dim, 
            max_actions=len(self.action_space_index.action_spaces)
            )

        # Positional encoding if not using ROPE/NOPE
        if not self.config.use_rope and not self.config.use_nope:
            self.positional_encoding = nn.Parameter(
                torch.randn(1, self.config.horizon, self.config.dit_dim) * 0.1)

        # DiT blocks
        self.dit = nn.ModuleList([
            FlowBlock(
                self.config.dit_dim, 
                self.config.n_heads,
                attn_pdrop=self.config.attn_pdrop,
                resid_pdrop=self.config.resid_pdrop,
                mlp_pdrop=self.config.mlp_pdrop,
                use_cross_attn=self.config.use_cross_attn,
                use_rope=self.config.use_rope,
                query_seq_len=self.config.query_seq_len,
                rope_theta=self.config.rope_theta,

            ) for _ in range(self.config.n_layers)
        ])

        # Create components per action space
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            input_dim = self.action_space_index.get_action_dim(action_idx)
            
            # Add encoder/decoder for this action
            self.action_encoders[action_name] = Mlp(
                in_features=input_dim, 
                hidden_features=self.config.dit_dim, 
                out_features=self.config.dit_dim, 
                bias=True
                ).to(self.device) 
            self.action_decoders[action_name] = nn.Linear(self.config.dit_dim, input_dim).to(self.device) 
                
            if self.config.action_type_adaln:
                self.adaln[action_name] = SharedAdaLNController(
                    self.config.dit_dim, 
                    global_conddim=self.config.dit_dim, 
                    use_cross_attn=self.config.use_cross_attn
                    ).to(self.device) 

            if self.config.use_proprio:
                # Add proprio encoder if needed for bimanual nav variant otherwise use zero encoder
                self.proprio_encoders[action_name] = Mlp(
                    input_dim, 
                    self.config.dit_dim, 
                    out_features=self.config.dit_dim, 
                    drop=0.2
                    ).to(self.device) 
            
    def _load_pretrained_weights(self, pretrained_model_path: str, mean_resizing: bool = False):
        """Loads pretrained weights, handling key mismatches (e.g., different prefixes)."""

        # init_vlm = self.vlm.language_model.model.encoder.layers[11].self_attn.out_proj.bias.clone()

        print(f"Loading pretrained weights from {pretrained_model_path}...")
        # Determine file type and load accordingly
        if pretrained_model_path.endswith('.safetensors'):
            # Load safetensors file
            from safetensors.torch import load_file
            state_dict = load_file(pretrained_model_path, device=str(self.device))
            checkpoint = {"state_dict": state_dict}  # Create checkpoint-like structure for compatibility
            print("Loaded safetensors file")
        else:
            # Load PyTorch checkpoint (.pt, .pth, .ckpt)
            checkpoint = torch.load(pretrained_model_path, map_location=self.device, weights_only=False)
            # checkpoint = torch.load(pretrained_model_path, map_location=self.device)
            # Extract the state dict (handle PyTorch Lightning or plain models)
            state_dict = checkpoint.get("state_dict", checkpoint)

        # Extract the state dict (handle PyTorch Lightning or plain models)
        state_dict = checkpoint.get("state_dict", checkpoint)

        if ("callbacks" in checkpoint and 
                "EMA" in checkpoint["callbacks"] and 
                "ema_weights" in checkpoint["callbacks"]["EMA"]):
                
                print("Found EMA weights in checkpoint, attempting to load them...")
                ema_weights_list = checkpoint['callbacks']['EMA']['ema_weights']
                
                # Get the original state dict to use as a reference for parameter names and shapes
                original_state_dict = checkpoint.get("state_dict", checkpoint)
                
                # Create a new state dict by matching EMA weights with original parameter names
                state_dict = {}
                ema_idx = 0
                
                for param_name, original_param in original_state_dict.items():
                    if ema_idx < len(ema_weights_list):
                        ema_weight = ema_weights_list[ema_idx]
                        
                        # Check if shapes match
                        if ema_weight.shape == original_param.shape:
                            state_dict[param_name] = ema_weight
                            ema_idx += 1
                        else:
                            # Shape mismatch - try to find the correct EMA weight by shape
                            found_match = False
                            for temp_idx in range(ema_idx, min(ema_idx + 20, len(ema_weights_list))):
                                if ema_weights_list[temp_idx].shape == original_param.shape:
                                    state_dict[param_name] = ema_weights_list[temp_idx]
                                    # Swap to maintain order
                                    ema_weights_list[temp_idx], ema_weights_list[ema_idx] = ema_weights_list[ema_idx], ema_weights_list[temp_idx]
                                    ema_idx += 1
                                    found_match = True
                                    break
                            
                            if not found_match:
                                # If no match found, use original parameter
                                print(f"Warning: No matching EMA weight found for {param_name}, using original")
                                state_dict[param_name] = original_param
                    else:
                        # No more EMA weights available, use original
                        print(f"Warning: Ran out of EMA weights at {param_name}, using original")
                        state_dict[param_name] = original_param
                
                print(f"Successfully matched {ema_idx} EMA weights out of {len(ema_weights_list)} total")

        # Fix key mismatches: remove 'agent.' prefix if it exists
        new_state_dict = {}
        # Handle language encoder/model naming mismatch
        for key, value in state_dict.items():
            new_key = key.replace("agent.", "")  # Remove 'agent.' if it exists
            
            # Handle language encoder/model naming mismatch
            if "vlm.language_encoder." in new_key:
                new_key = new_key.replace("vlm.language_encoder.", "vlm.language_model.model.encoder.")
            #elif "vlm.language_model." in new_key and "vlm.language_model.model." not in new_key:
                # If it's already language_model but missing the nested structure, add it
                #new_key = new_key.replace("vlm.language_model.", "vlm.language_model.model.encoder.")
            # Handle MLP naming mismatch
            new_key = new_key.replace(".mlp.c_fc1.", ".mlp.fc1.")
            new_key = new_key.replace(".mlp.c_fc2.", ".mlp.fc2.")
            new_key = new_key.replace(".mlp.c_proj.", ".mlp.proj.")
            new_state_dict[new_key] = value

        # 创建新分支会和当前模型的参数名冲突：
        current_state_dict = self.state_dict()
        filtered_state_dict = {}
        for key, value in new_state_dict.items():
            if key in current_state_dict:
                # 检查形状是否匹配
                if current_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    print(f"⚠️ 跳过形状不匹配的参数: {key} | 预训练模型: {value.shape} | 当前模型: {current_state_dict[key].shape}")
            else:
                # 只保留当前模型分支的参数
                print(f"🫥 跳过当前模型不存在的参数: {key}")
        
        # 加载过滤后的状态字典
        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
        # Log mismatches for debugging
        print(f"Pretrained weights loaded with the following issues:")
        print(f"成功加载 {len(filtered_state_dict)}/{len(new_state_dict)} 个参数")
        print(f"⚠️ Missing keys: {len(missing_keys)}")
        if missing_keys:
            print(f"  ⚠️ Missing keys (not found in checkpoint, using default init): {len(missing_keys)}")
            print(f"    {missing_keys[:30]} ...")  # Show first 30 for brevity
        print(f"🫥 Unexpected keys: {len(unexpected_keys)}")
        if unexpected_keys:
            print(f"  🫥 Unexpected keys (ignored): {len(unexpected_keys)}")
            print(f"    {unexpected_keys[:30]} ...")  # Show first 30 for brevity
        if not missing_keys and not unexpected_keys:
            print("  ✅ All keys matched successfully!") 

        # 检查vlm是否被更新
        # final_vlm = self.vlm.language_model.model.encoder.layers[11].self_attn.out_proj.bias
        # if torch.equal(final_vlm, init_vlm):
        #     print("❌ VLM参数没有被更新")
        # else:
        #     print("✅ VLM参数被成功更新")

        return missing_keys, unexpected_keys

    def _configure_optimizers(self, optimizer_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        no_decay = ['bias', 'LayerNorm', 'layernorm', 'ln', 'norm']
        decay_group = []
        no_decay_group = []
        vlm_params = set(p for p in self.vlm.parameters())
        for name, param in self.named_parameters():
            if param.requires_grad and param.is_leaf and param not in vlm_params:
                if any(nd in name.lower() for nd in no_decay):
                    no_decay_group.append(param)
                else:
                    decay_group.append(param)
        dit_optim_groups = [
            {"params": decay_group, "weight_decay": optimizer_config["transformer_weight_decay"]},
            {"params": no_decay_group, "weight_decay": 0.0}
        ]
        vlm_optim_params = [p for p in self.vlm.parameters() if p.requires_grad]
        return dit_optim_groups, vlm_optim_params

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        # Sample prior.
        z = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, 16),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )
         # Integration
        dt = 1.0 / self.num_inference_steps
        dt_tensor = torch.tensor([dt] * batch_size, device=device).view([batch_size] + [1]*(z.dim()-1))

        for i in range(self.num_inference_steps, 0, -1):
            t_val = i / self.num_inference_steps
            t_tensor = torch.full((batch_size,), t_val, device=device)

            # Predict velocity field
            vc, _ = self.dit_forward(z, t_tensor, cond)
            z = z - dt_tensor * vc
        
        sample = z.clamp(-1, 1)
        return sample

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        cond = self.encode_observations(batch)

        # run sampling
        actions = self.conditional_sample(batch_size, cond=cond, noise=noise)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, n_obs_steps, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps
        
        # cond:
        cond = self.encode_observations(batch)
        # Forward rectified flow.
        trajectory = batch[ACTION]
        b = trajectory.shape[0]
        device = trajectory.device
        default_dtype = trajectory.dtype

        # Sample time based on sampling strategy
        if self.config.sampling_type == "pi_zero":
            alpha, beta = 1.5, 1.0
            t = torch.distributions.Beta(alpha, beta).sample((b,)).to(device)
            t = t.clamp(max=0.999)
        elif self.config.sampling_type == "ln":
            t = torch.sigmoid(torch.randn((b,), device=device))
            t = t.clamp(max=0.999).to(default_dtype)
        elif self.config.sampling_type == "uniform":
            eps = 1e-5
            t = (torch.rand(1, device=device) + torch.arange(b, device=device) / b) % (1 - eps)
            t = t.to(default_dtype)
        else:
            raise NotImplementedError(f"Sampling type {self.sampling_type} not implemented")

        # Interpolate between actions and noise
        texp = t.view([b] + [1] * (trajectory.dim() - 1))
        # z1 = torch.randn_like(trajectory, device=device).to(default_dtype)
        z1 = torch.zeros_like(trajectory)
        action_type = cond['action_type']
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                noise_slice = torch.randn(
                    (mask.sum(), trajectory.size(1), adim), 
                    dtype=default_dtype, 
                    device=device
                    )
                z1[mask, :, :adim] = noise_slice
        zt = (1 - texp) * trajectory + texp * z1

        # Forward pass
        vtheta, _ = self.dit_forward(zt, t, cond)
        
        # valid_mask
        valid_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                mask = batch['valid']  # 在这里应用valid，处理错误数据
                adim = self.action_space_index.get_action_dim(action_idx)
                mask_expanded = mask.view(-1, 1, 1).expand(-1, trajectory.size(1), adim).to(device)
                valid_mask[mask, :, :adim] = mask_expanded[mask]
        
        # Compute loss on valid dimensions only
        diff = (z1 - trajectory) - vtheta
        valid_diff = torch.where(
            valid_mask, 
            diff, 
            torch.tensor(0.0, device=diff.device)
            ) # valid_diff = diff[valid_mask]  # valid_diff = diff
        # loss = (valid_diff ** 2)  # l2
        loss = torch.abs(valid_diff)  # l1
        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            # loss = loss * in_episode_bound.unsqueeze(-1)
            in_episode_bound = in_episode_bound.unsqueeze(-1).expand(*in_episode_bound.shape, loss.size(-1))
            valid_mask = valid_mask & in_episode_bound
            loss = loss * in_episode_bound
        loss_mean = loss.sum() / (valid_mask.sum().float() + 1e-8)
        return loss_mean

    def encode_observations(self, batch):
        """Encode observations using Florence-2"""
        
        device = get_device_from_parameters(self)  # device = self.device
        default_dtype = get_dtype_from_parameters(self)  # next(self.parameters()).dtype
        
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        # Extract visual features
        # 根据flower的实现用同一个vlm的encoder分别编码
        images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
        img_features_list = torch.cat([
            self.vlm._encode_image(images) for images in images_per_camera
            ])
        img_features = einops.rearrange(
            img_features_list, "(n b s) c dim -> b (s n c) dim", b=batch_size, s=n_obs_steps
            )
        
        # Get text embeddings
        # Get text embeddings once to reuse
        # constructed_prompts, batch_action_index = self.construct_prompts(batch)
        # text_embeds, txt_attention_mask = self._get_text_embeddings(constructed_prompts, device)
        batch_action_index = batch['action_index'].to(device)
        text_embeds = self._get_text_embeddings_new(batch['text_input_ids'], device)
        txt_attention_mask = batch['text_attention_mask'].to(device)
        # Add task prompt and aggregation tokens
        task_prompt = self.prompt_embeds.expand(batch_size, -1, -1)
        
        # Merge sequence
        merged_embeds = torch.cat([
            task_prompt.to(img_features.device),
            img_features,
            text_embeds.to(img_features.device)
        ], dim=1)

        # Create attention mask
        # attention_mask = torch.ones(merged_embeds.shape[:2], device=merged_embeds.device)
        prompt_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        txt_attention_mask = txt_attention_mask.to(device).squeeze(1)  # get attention mask from txt
        vis_attention_mask = torch.ones(img_features.shape[:2], device=device)  # define attention mask for image
        attention_mask = torch.cat([prompt_mask, vis_attention_mask, txt_attention_mask], dim=1)
        # Process through encoder
        features = self.vlm.get_encoder()(
            inputs_embeds=merged_embeds,
            attention_mask=attention_mask
        ).last_hidden_state

        # Apply dropout 
        features = self.vlm_token_dropout(features)

        # Prepare frequency and action space embeddings
        frequency_embeds = self.frequency_embedder(
            torch.ones(batch_size, 1, 1).to(device) * self.config.data_frequency
        )  # 暂时固定
        
        # Get proprioception if enabled
        proprio = None
        if self.config.use_proprio:
            proprio = batch['observation.state'].to(device).to(default_dtype)

        return {
            'features': features,
            'frequency_embeds': frequency_embeds,
            'action_space_embeds': self.action_space_embedder(batch_action_index.to(device)),
            'action_type': batch_action_index.to(device),
            'proprio': proprio,
            'attention_mask': attention_mask,
        }
    
    def dit_forward(self, z: torch.Tensor, t: torch.Tensor, cond_dict: dict) -> torch.Tensor:
        """
        Forward pass through the DiT blocks.
        """
        device = get_device_from_parameters(self)  # device = self.device
        default_dtype = get_dtype_from_parameters(self)  # next(self.parameters()).dtype

        # Get conditioning information
        cond = cond_dict['features'].to(default_dtype)
        frequency_embeds = cond_dict['frequency_embeds'].squeeze(1).to(default_dtype)
        action_type = cond_dict['action_type'].to(device)
        
        # Handle proprioception
        if self.config.use_proprio and cond_dict['proprio'] is not None:
            # 这里处理为每个时间步的proprioception取平均值, flower本身不支持n_obs
            proprio = cond_dict['proprio'].to(default_dtype)
            proprio_embeds = self.encode_proprio(proprio, action_type, frequency_embeds.shape)
        else:
            proprio_embeds = torch.zeros_like(frequency_embeds)

        # Encode actions
        z, valid_dims = self.encode_actions(z, action_type)
        
        # Add positional encoding if not using ROPE/NOPE
        if not self.config.use_rope and not self.config.use_nope:
            z = z + self.positional_encoding

        # Process embeddings
        t_emb = stateless_norm(self.t_embedder(t)) + \
                stateless_norm(frequency_embeds).squeeze(1) + \
                stateless_norm(proprio_embeds).squeeze(1)
        
        cond = self.cond_linear(self.cond_norm(cond))
        
        # Set up conditioning
        if self.config.use_adaln_cond:
            vlm_token = cond[:, 0, :] if self.config.use_readout_token else cond.mean(dim=1)
            global_cond = vlm_token + t_emb
        else:
            global_cond = t_emb
        
        # Setup context
        cx = z
        context = cond if self.config.use_cross_attn else None
        
        # Get adaln signals
        if not self.config.action_type_adaln:
            global_adaln = self.adaln(global_cond)
        else:
            global_adaln = self.action_specific_adaln(global_cond, action_type)

        # Process through DiT blocks
        # print(next(self.vlm.parameters()).device)
        # print(next(self.dit.parameters()).device)
        # import pdb; pdb.set_trace()
        for layer in self.dit:
            cx = layer(
                cx, 
                global_cond, 
                context=context, 
                is_causal=True, 
                global_adaln=global_adaln
            )
            
        # Decode and return
        return self.decode_actions(cx, action_type, valid_dims), cx

    def _create_prompt_embed(self, prompt_text):
        """Create embeddings for prompt tokens"""
        # Add special token if not in vocabulary
        self.tokenizer.add_special_tokens({'additional_special_tokens': [prompt_text]})
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        
        # Get token ID and create embedding
        prompt_token_id = self.tokenizer.convert_tokens_to_ids(prompt_text)
        prompt_embed = nn.Parameter(
            self.vlm.get_input_embeddings()(torch.tensor(prompt_token_id)), 
            requires_grad=False
        )
    
        return prompt_embed.unsqueeze(0).unsqueeze(0)

    def construct_prompts(self, dataset_batch):
        """
        Constructs prompts for Florence-2's encoder to extract task-relevant visual features.
        
        Args:
            dataset_batch: Dictionary containing task information including language instructions
            
        Returns:
            text_prompts: List of formatted prompts for encoder conditioning
        """
    
        language_instruction = dataset_batch["task"]
        text_prompts = []
        batch_action_index = []
        for idx, instruction in enumerate(language_instruction):
            if self.config.vlm_prompt_style == "default":
                # Original instruction only
                robot_type = dataset_batch['info']["robot_type"][idx]
                action_index = self.action_space_index.robot_mapping[robot_type]
                batch_action_index.append(action_index)
                instruction = generate_policy_prompt(
                    instruction,
                    robot_name=robot_type,
                    num_arms=self.action_space_index.get_num_arms(action_index),
                    action_space=f"{self.action_space_index.get_action_dim(action_index)}D continuous",
                    prompt_style="minimal",
                    include_meta=True
                    )
                text_prompts.append(instruction)
                # text_prompts.append(self.format_instruction(instruction))
            else:
                raise ValueError(f"Unknown prompt style: {self.config.vlm_prompt_style}")
        
        batch_action_index = torch.tensor(batch_action_index)
        return text_prompts, batch_action_index

    def _get_text_embeddings_new(self, text_inputs, device):
        """Get text embeddings to use with VLM"""
        text_inputs = text_inputs.to(device)
        text_embeds = self.vlm.get_input_embeddings()(text_inputs)
        return text_embeds
    
    def encode_proprio(self, proprio: torch.Tensor, action_type: torch.Tensor, output_shape) -> torch.Tensor:
        """
        Encode proprioception based on action type.
        """
        batch_size, _, _ = output_shape
        device = get_device_from_parameters(self)
        default_dtype = next(self.parameters()).dtype
        
        if not self.config.use_proprio:
            return torch.zeros(batch_size, self.config.dit_dim, device=device)

        # proprio = proprio.squeeze(1).to(device)
        proprio = proprio.mean(dim=1).to(device)
        encoded_proprio = torch.zeros(batch_size, self.config.dit_dim, device=device, dtype=DEFAULT_DTYPE)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                encoded_proprio[mask] = self.proprio_encoders[action_name](proprio[mask, :adim]).squeeze(1)
        return encoded_proprio
    
    def encode_actions(self, z: torch.Tensor, action_type: torch.Tensor) -> torch.Tensor:
        """Encode actions using action-specific encoders."""
        batch_size, _, _ = z.shape
        device = get_device_from_parameters(self)
        default_dtype = next(self.parameters()).dtype
        
        # for action_name, action_idx in self.action_space_index.action_spaces.items():
        #     mask = (action_type == action_idx)
        #     if mask.any():
        #         encoded = self.action_encoders[action_name](z)
        
        encoded = torch.zeros(batch_size, z.shape[1], self.config.dit_dim, device=device, dtype=DEFAULT_DTYPE)
        valid_dims = torch.zeros_like(z, dtype=default_dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                valid_dims[mask, :, :adim] = 1
                encoded[mask] = self.action_encoders[action_name](z[mask, :, :adim])
        
        return encoded, valid_dims

    def decode_actions(self, z: torch.Tensor, action_type: torch.Tensor, valid_dims: torch.Tensor) -> torch.Tensor:
        """Decode actions using action-specific decoders."""
        device = get_device_from_parameters(self)
        default_dtype = next(self.parameters()).dtype
        
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                decoded = self.action_decoders[action_name](z)

        batch_size = z.shape[0]
        max_action_dim = self.action_space_index.get_max_action_dim()
        decoded = torch.zeros(batch_size, z.shape[1], max_action_dim, device=device, dtype=DEFAULT_DTYPE)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                pred = self.action_decoders[action_name](z[mask])
                decoded[mask, :, :adim] = (pred[..., :adim] * valid_dims[mask, :, :adim]).to(DEFAULT_DTYPE)

        return decoded

    def action_specific_adaln(self, global_cond: torch.Tensor, action_type: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate action-specific AdaLN signals.
        """
        device = get_device_from_parameters(self)  # global_cond.device
        default_type = next(self.parameters()).dtype
        batch_size = global_cond.shape[0]
        num_chunks = 9 if self.config.use_cross_attn else 6
        
        mod_signals = [
            torch.zeros(batch_size, self.config.dit_dim, device=device, dtype=DEFAULT_DTYPE) 
            for _ in range(num_chunks)
        ]
        
        # for action_idx in range(len(self.action_space_index.action_spaces)):
        #     mask = (action_type == action_idx)
        #     if mask.any():
        #         action_name = self.action_space_index.get_action_name(action_idx)
        #         action_mod = self.adaln[action_name](global_cond)
        #         for i, signal in enumerate(action_mod):
        #             mod_signals[i] = signal

        for action_idx in range(len(self.action_space_index.action_spaces)):
            mask = (action_type == action_idx)
            if mask.any():
                action_name = self.action_space_index.get_action_name(action_idx)
                action_mod = self.adaln[action_name](global_cond[mask])
                for i, signal in enumerate(action_mod):
                    mod_signals[i][mask] = signal
        return mod_signals


if __name__ == "__main__":
    from lerobot.policies.flower.configuration_flower import FlowerConfig
    libero_weights = "/mnt/data/daiwanqin/models/flower_train/22-48-39/seed_42/saved_models/epoch=49_eval_lh/avg_seq_len=0.93.ckpt"
    config = FlowerConfig()
    config.use_proprio = False
    model = FlowerModel(config)
    model._load_pretrained_weights(libero_weights)

