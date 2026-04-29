# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
====== ray_trainer.py 整体架构 ======

这是 PPO 训练的核心控制器，包含：

1. ResourcePoolManager：GPU 资源池管理
2. apply_kl_penalty()：KL 惩罚计算
3. compute_advantage()：Advantage 计算（GAE/GRPO）
4. RayPPOTrainer：PPO 训练主类

====== RayPPOTrainer 核心功能 ======

RayPPOTrainer 负责：
1. 创建和管理 WorkerGroup（Actor, Critic, RM, Ref）
2. 执行 9 步训练循环
3. 处理 checkpoint 和 validation

====== PPO 9 步训练循环 ======

Step 1: generate_sequences()      # Actor 生成 response
Step 2: compute_reward()           # 计算 reward score
Step 3: compute_advantage()        # 计算 advantage（GAE/GRPO）
Step 4: update_critic()            # 更新 Critic（如果需要）
Step 5: update_actor()             # 更新 Actor（PPO clip）
Step 6: validate()                 # 验证（可选）
Step 7: save_checkpoint()          # 保存 checkpoint（可选）
Step 8: log_metrics()              # 记录 metrics
Step 9: repeat for next batch      # 循环下一个 batch

====== 数据流示意 ======

DataLoader → batch (DataProto)
    ↓
ActorWorker.generate_sequences() → batch + responses
    ↓
RewardModel.compute_reward() → batch + token_level_scores
    ↓
CriticWorker.compute_values() → batch + values
    ↓
compute_advantage() → batch + advantages
    ↓
ActorWorker.update_policy() → loss
    ↓
CriticWorker.update_value() → loss

PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.

    ====== ResourcePoolManager 的作用 ======

    管理 GPU 资源池：
    1. 创建 Ray PlacementGroup（GPU 资源调度单位）
    2. 将 Role 映射到对应的资源池
    3. 检查资源是否满足需求

    ====== 数据结构示例 ======

    resource_pool_spec (输入):
    {
        "global_pool": [8, 8],  # 两个节点各 8 GPU
        "reward_pool": [4],     # 单节点 4 GPU
    }

    mapping (输入):
    {
        Role.ActorRollout: "global_pool",
        Role.Critic: "global_pool",
        Role.RewardModel: "reward_pool",
    }

    resource_pool_dict (输出):
    {
        "global_pool": RayResourcePool(process_on_nodes=[8, 8]),
        "reward_pool": RayResourcePool(process_on_nodes=[4]),
    }

    ====== max_colocate_count 的含义 ======

    max_colocate_count: GPU 复用次数
    - 1: 每个 WorkerGroup 独占 GPU（FSDP 默认）
    - 3: 一个 GPU 可分配给 3 个 WorkerGroup（Actor, Critic, Ref 共享）

    ====== 使用流程 ======

    1. main_ppo.py 创建 resource_pool_spec 和 mapping
    2. RayPPOTrainer.init_workers() 调用 create_resource_pool()
    3. RayResourcePool 创建 PlacementGroup
    4. WorkerGroup 从资源池获取 GPU
    """

    # resource_pool_spec: 资源池配置
    # 格式：{pool_name: [n_gpus_per_node] * nnodes}
    # 例如：{"global_pool": [8, 8]} 表示两个节点各 8 GPU
    resource_pool_spec: dict[str, list[int]]

    # mapping: Role → 资源池名称的映射
    # 例如：{Role.ActorRollout: "global_pool"}
    mapping: dict[Role, str]

    # resource_pool_dict: 创建后的资源池实例
    # key: pool_name, value: RayResourcePool 实例
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        ====== 创建流程 ======

        1. 遍历 resource_pool_spec 中的每个资源池
        2. 创建 RayResourcePool 实例
        3. 调用 RayResourcePool.get_placement_groups() 创建 PlacementGroup
        4. 检查资源是否满足需求

        ====== PlacementGroup 创建示意 ======

        resource_pool_spec = {"global_pool": [8, 8]}
        → RayResourcePool(process_on_nodes=[8, 8])
        → PlacementGroup with 16 bundles (每个 bundle = 1 GPU + N CPU)

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models

            # RayResourcePool: GPU 资源池管理类
            # process_on_nodes: 每个节点的进程数（通常等于 GPU 数）
            # use_gpu=True: 分配 GPU 资源
            # max_colocate_count=3: 一个 GPU 可分给 3 个 WorkerGroup
            # name_prefix: 资源池名称前缀（用于 PlacementGroup 命名）
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )

            # 存储资源池实例
            self.resource_pool_dict[resource_pool_name] = resource_pool

        # 检查 Ray 集群是否有足够的 GPU
        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls.

        ====== 使用示例 ======

        role = Role.ActorRollout
        mapping[role] = "global_pool"
        → 返回 resource_pool_dict["global_pool"]

        Args:
            role: Worker 的角色类型（如 ActorRollout, Critic）

        Returns:
            RayResourcePool: 对应角色的资源池实例
        """
        # self.mapping[role]: 获取 Role 对应的资源池名称
        # self.resource_pool_dict[name]: 获取资源池实例
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster.

        ====== 计算示例 ======

        resource_pool_spec = {"global_pool": [8, 8], "reward_pool": [4]}
        → sum([8, 8, 4]) = 20 GPU

        Returns:
            int: 所有资源池的 GPU 总数
        """
        # 遍历所有资源池，累加 GPU 数量
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster.

        ====== 检查逻辑 ======

        1. 获取 Ray 集群中每个节点的可用 GPU 数
        2. 计算总可用 GPU 数
        3. 计算总需求 GPU 数
        4. 如果可用 < 需求，抛出异常

        ====== 异常示例 ======

        需求 20 GPU，只有 16 GPU 可用：
        → ValueError("Total available GPUs 16 is less than total desired GPUs 20")

        ====== NPU 支持 ======

        如果 GPU 不可用，检查 NPU（Ascend 芯片）
        """
        # ray._private.state.available_resources_per_node(): 获取每个节点的可用资源
        node_available_resources = ray._private.state.available_resources_per_node()

        # 提取每个节点的可用 GPU/NPU 数量
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        # 计算总可用 GPU 数
        total_available_gpus = sum(node_available_gpus.values())

        # 计算总需求 GPU 数（从 resource_pool_spec）
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )

        # 检查资源是否足够
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    ====== KL Penalty 的作用 ======

    KL Penalty 用于约束策略不偏离参考策略太多：
    reward = task_reward - β * KL(π || π_ref)

    防止策略过度优化导致：
    - 生成内容质量下降
    - 策略偏离原始模型太远

    ====== KL 计算公式 ======

    KL(π || π_ref) = Σ π(x) * log(π(x) / π_ref(x))
    在实践中用：KL = log_prob_current - log_prob_ref

    ====== 数据流示例 ======

    输入 data.batch:
    - old_log_probs: [bsz, seq_len]      # 当前策略的 log probability
    - ref_log_prob: [bsz, seq_len]       # 参考策略的 log probability
    - token_level_scores: [bsz, seq_len] # 原始 reward score
    - response_mask: [bsz, seq_len]      # 1=response token, 0=prompt token

    计算 kld:
    - kld = old_log_probs - ref_log_prob  # KL divergence
    - kld = kld * response_mask           # 只对 response 计算

    输出 data.batch:
    - token_level_rewards: [bsz, seq_len] # reward - β * KL

    ====== Adaptive KL Controller ======

    kl_ctrl.value: KL 系数 β
    - 动态调整：如果 KL 太大，增大 β（加强约束）
    - 如果 KL 太小，减小 β（放松约束）

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    # response_mask: 只对 response token 计算 KL（不对 prompt）
    # 1 表示 response token，0 表示 prompt token
    response_mask = data.batch["response_mask"]

    # token_level_scores: 原始 reward score（来自 Reward Model 或规则）
    token_level_scores = data.batch["token_level_scores"]

    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.

    # kl_penalty(): 计算 KL divergence
    # kld = old_log_probs - ref_log_prob
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)

    # 只对 response token 计算 KL（乘以 mask）
    kld = kld * response_mask

    # β: KL 系数，控制 KL 惩罚的强度
    beta = kl_ctrl.value

    # token_level_rewards: 最终的 token-level reward
    # reward = task_reward - β * KL
    token_level_rewards = token_level_scores - beta * kld

    # 计算当前 batch 的平均 KL（用于自适应调整 β）
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    # kl_ctrl.update(): 自适应调整 KL 系数 β
    # - 如果 current_kl > target_kl，增大 β
    # - 如果 current_kl < target_kl，减小 β
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)

    # 将 token_level_rewards 存入 data
    data.batch["token_level_rewards"] = token_level_rewards

    # 记录 KL 相关 metrics
    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    ====== response_mask 的作用 ======

    response_mask 用于区分 prompt 和 response：
    - prompt token: 0（不参与某些计算）
    - response token: 1（参与计算）

    用途：
    1. 计算 reward 只对 response
    2. 计算 KL 只对 response
    3. 计算 loss 只对 response

    ====== 数据流示例 ======

    输入 data.batch:
    - attention_mask: [bsz, seq_len]  # prompt + response 的 mask
    - responses: [bsz, response_len]  # 只有 response 部分

    假设:
    - seq_len = 512 (prompt 256 + response 256)
    - response_len = 256

    输出:
    - response_mask: [bsz, 256]  # 只取后 256 个位置

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    # responses: response 部分的 token ids
    responses = data.batch["responses"]

    # response_length: response 的长度（token 数）
    response_length = responses.size(1)

    # attention_mask: 整个序列的 mask（prompt + response）
    attention_mask = data.batch["attention_mask"]

    # 只取 attention_mask 的后 response_length 个位置
    # [:, -response_length:] 表示从倒数第 response_length 个位置开始取
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    ====== Advantage 的作用 ======

    Advantage 估计"某个动作比平均好多少"：
    A(s, a) = Q(s, a) - V(s)

    用于 PPO policy update：
    ratio = π(a|s) / π_old(a|s)
    loss = -min(ratio * A, clip(ratio) * A)

    ====== Advantage 计算方法 ======

    | adv_estimator | 方法 | 需要 Critic |
    |---------------|------|------------|
    | gae | Generalized Advantage Estimation | 是 |
    | grpo | Group Relative Policy Optimization | 否 |
    | reinforce | REINFORCE（无 baseline） | 否 |
    | reinforce_plus_plus | REINFORCE++（有 baseline） | 否 |

    ====== GAE 计算公式 ======

    A_t = Σ (γλ)^l * (r_l + γV(s_{l+1}) - V(s_l))

    参数：
    - gamma (γ): 折扣因子，未来 reward 的权重
    - lam (λ): GAE 平滑参数，平衡短期和长期 advantage

    ====== GRPO 计算公式 ======

    对于同一 prompt 的 n 个 response：
    A = (r - mean) / std

    不需要 Critic，用 group 的 mean/std 代替。

    ====== 数据流示例 ======

    输入 data.batch:
    - token_level_rewards: [bsz, seq_len]  # 每个 token 的 reward
    - values: [bsz, seq_len]               # Critic 预测的价值（GAE 需要）
    - response_mask: [bsz, seq_len]        # response mask

    输出 data.batch:
    - advantages: [bsz, seq_len]           # 每个 token 的 advantage
    - value_targets: [bsz, seq_len]        # Critic 的目标值（用于更新）

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    # 如果 data 中没有 response_mask，计算它
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)

    # ====== GAE Advantage 计算 ======
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        # compute_gae_advantage_return(): 计算 GAE advantage 和 return
        # 输入：
        # - token_level_rewards: [bsz, seq_len] 每个 token 的 reward
        # - values: [bsz, seq_len] Critic 预测的价值
        # - response_mask: [bsz, seq_len] 只对 response 计算
        # 输出：
        # - advantages: [bsz, seq_len] GAE advantage
        # - returns: [bsz, seq_len] Critic 的目标值
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

        # PF-PPO: Preference Fine-tuned PPO（可选）
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )

    # ====== GRPO Advantage 计算 ======
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        # compute_grpo_outcome_advantage(): 计算 GRPO advantage
        # 输入：
        # - token_level_rewards: [bsz, seq_len] 每个 token 的 reward
        # - response_mask: [bsz, seq_len] 只对 response 计算
        # - index (uid): [bsz] 用于分组（同一 prompt 的 response 共享 uid）
        # 输出：
        # - advantages: [bsz, seq_len] GRPO advantage
        # - returns: [bsz, seq_len] 与 advantages 相同（GRPO 不需要 Critic）
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    # ====== 其他 Advantage Estimator ======
    else:
        # handle all other adv estimator type other than GAE and GRPO
        # get_adv_estimator_fn(): 根据 adv_estimator 获取计算函数
        # 支持：REINFORCE, REINFORCE++, rloo, is_qa 等
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)

        # 构建参数字典
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }

        # uid: 用于分组（可选）
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]

        # reward_baselines: 用于 baseline 计算（可选）
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        # adv_estimator_fn(): 计算 advantage 和 returns
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    ====== RayPPOTrainer 的作用 ======

    RayPPOTrainer 是 PPO 训练的核心控制器：
    1. 创建和管理 WorkerGroup（Actor, Critic, RM, Ref）
    2. 执行 9 步训练循环
    3. 处理 checkpoint 和 validation

    ====== RayPPOTrainer 运行位置 ======

    RayPPOTrainer 运行在 Driver 进程（TaskRunner Actor）：
    - Driver 进程不占用 GPU（只做调度）
    - Worker 进程占用 GPU（做实际计算）

    ====== WorkerGroup 结构 ======

    RayPPOTrainer 管理多个 WorkerGroup：
    - ActorRolloutWG: Actor + Rollout（生成 response）
    - CriticWG: Critic（计算 value）
    - RewardModelWG: Reward Model（计算 reward）
    - RefPolicyWG: Reference Policy（计算 KL）

    每个 WorkerGroup 包含多个 Worker：
    - WorkerGroup: 管理多个 Worker 的调度器
    - Worker: Ray Actor，独立进程执行任务

    ====== PPO 9 步训练循环（fit() 方法）======

    Step 1: generate_sequences()      # Actor 生成 response
    Step 2: compute_reward()           # 计算 reward score
    Step 3: compute_advantage()        # 计算 advantage（GAE/GRPO）
    Step 4: update_critic()            # 更新 Critic（如果需要）
    Step 5: update_actor()             # 更新 Actor（PPO clip）
    Step 6: validate()                 # 验证（可选）
    Step 7: save_checkpoint()          # 保存 checkpoint（可选）
    Step 8: log_metrics()              # 记录 metrics
    Step 9: repeat for next batch      # 循环下一个 batch

    ====== 数据流示意 ======

    DataLoader → batch (DataProto)
        ↓
    ActorWorker.generate_sequences() → batch + responses
        ↓
    RewardModel.compute_reward() → batch + token_level_scores
        ↓
    CriticWorker.compute_values() → batch + values
        ↓
    compute_advantage() → batch + advantages
        ↓
    ActorWorker.update_policy() → loss
        ↓
    CriticWorker.update_value() → loss

    ====== 关键属性 ======

    | 属性 | 含义 |
    |------|------|
    | role_worker_mapping | Role → WorkerGroup 实例 |
    | resource_pool_manager | GPU 资源池管理 |
    | tokenizer | 文本编码器 |
    | reward_fn | reward 计算函数 |
    | train_dataset | 训练数据集 |

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.

        ====== 初始化流程 ======

        __init__() 只存储配置和参数，不创建 WorkerGroup。
        WorkerGroup 在 init_workers() 中创建。

        ====== role_worker_mapping ======

        role_worker_mapping 是 Role → WorkerClass 的映射：
        {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RewardModel: ray.remote(RewardModelWorker),
        }

        在 init_workers() 中转换为 Role → WorkerGroup 实例。

        ====== 参数说明 ======

        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        return_dict: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor | dict[str, Any]:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            return_dict: Whether to return dict format with reward_extra_info (for validation)
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
            If return_dict=False and sum_reward=True: summed reward_tensor (1D tensor)
            If return_dict=False and sum_reward=False: reward_tensor (2D tensor)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if return_dict:
                # Extract reward_extra_info if available
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                # If sum_reward=True, only return tensor (for REMAX baseline)
                if sum_reward:
                    return reward_tensor
                # Otherwise, return tuple with reward_extra_info (for training loop)
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_infos_dict = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return reward_tensor, reward_extra_infos_dict

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if return_dict:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_info = result.get("reward_extra_info", {})
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            # 把每个 batch 包装成 VeRL 的统一数据容器 DataProto，方便后续在 Controller 和 Worker 之间传递
            # DataProto 包含 batch（张量字典，如 prompts, attention_mask）和 non_tensor_batch（非张量字典，如字符串、元数据）
            test_batch = DataProto.from_single_dict(test_data)
            # 确保每个样本有唯一uuid
            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )
            # repeat_times=n：每条 prompt 复制 n 份。这样同一个问题能生成 n 个不同回答，用于计算 pass@n、maj@n 等指标。
            #
            # interleave=True：保证同一个 prompt 的副本是连续排列的（如 [A1, A2, A3, B1, B2, B3] 而不是 [A1, B1, A2, B2, A3, B3]），方便后面按组处理。
            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )
            # 如果启用了模型型 RM（即用神经网络打分），且这个数据集的第一个样本标明“需要模型打分”，则直接返回空字典。
            # 原因：验证时通常只用规则型 RM（如答案比对），因为模型型 RM 推理成本高，且验证阶段可能还没加载 RM。
            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}
            # 从每个样本的 non_tensor_batch 里取出正确答案（ground_truth），用于后续 dump 和可能的指标计算。
            # 全部追加到全局列表 sample_gts。
            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)
            # 从完整的 test_batch 中提取出生成所需的字段（通常包括 prompts, attention_mask 等），创建专门用于远程调用的 DataProto。
            # 这个方法内部会去掉不需要的信息，减轻传输负担。
            test_gen_batch = self._get_gen_batch(test_batch)
            # 将控制参数打包进 meta_info，这些信息会被传递到远端的 Actor Worker，告诉它：“这是验证模式，不要算 log prob，是否需要随机采样，以及当前步数”。
            # validate=True 是区分验证和训练 Rollout 的关键标志。
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            # 计算并行计算单元总数（同步模式下是 WorkerGroup 的世界大小，异步模式下是 agent 的 worker 数量）。
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            # 用 pad_dataproto_to_divisor 把 batch 大小补齐到该数的整数倍，避免 Tensor Parallelism 时维度不对齐。
            # 返回补齐后的 batch 和补了多少样本。
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            # 同步模式：Controller 调用 actor_rollout_wg.generate_sequences，阻塞等待所有 Worker 生成完毕。
            # 异步模式：Controller 通过 async_rollout_manager 发出任务，不等待，稍后取结果。
            # 这里就是 Controller（单控制器） 指挥 Worker（SPMD） 执行自回归生成（Generation）的精确位置。生成的 responses 会放在返回的 DataProto 的 batch["responses"] 里。
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")
            # 从 batch 中取出响应的 token id (responses)。
            # 用 tokenizer 解码成可读文本，并加入 sample_outputs 全局列表
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            # 把生成的 responses 合并回原来的 test_batch（现在 test_batch 同时有 prompts 和 responses）。
            # 再次打上 validate=True 标记，防止后续处理误判。
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
            # 解码原始 prompt，收集到 sample_inputs。
            # 同时收集 uid。
            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])
            # 调用你逐行看过的那 _compute_or_extract_reward，使用验证专用奖励函数 val_reward_fn，返回字典。
            # 从结果字典中取出 reward_tensor，形状 (batch_size, num_reward_dims)。 sum(-1) 沿最后一维求和，得到每个样本的总分（标量），存入 scores 列表。
            # 这些分数会被追加到 sample_scores 全局列表。
            # evaluate using reward_function
            result = self._compute_or_extract_reward(test_batch, reward_fn=self.val_reward_fn, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            # 首先把总分加入 reward 键。
            reward_extra_infos_dict["reward"].extend(scores)
            # 如果有额外信息（例如 acc 准确率、format_score 格式分），逐一追加到全局字典的对应列表中。
            # 处理了 numpy 数组和普通列表两种情况。
            reward_extra_info = result.get("reward_extra_info", {})
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])
            # 每个样本可能标记了来自哪个数据集（如 "math"），收集起来用于最后的分组指标计算。
            # 如果没有标记，就用 "unknown" 填充。
            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
        # 如果配置了 wandb 等日志工具，把输入、输出、分数记录成表格，方便可视化查看模型输出变化。
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        # 如果配置了 validation_data_dir，就把所有样本的输入、输出、ground truth、分数、额外信息存成 JSON/CSV 文件，供离线分析。
        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )
        # 确保所有奖励额外信息的长度与总分列表一致（要么为空，要么长度相同），否则报错。
        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)
        # process_validation_metrics：根据 data_source 和 uid 对样本分组，计算各种指标（如 mean@1、maj@4、best@4）
        # 返回嵌套字典：数据来源 -> 变量名 -> 指标名 -> 值。
        # 随后遍历所有指标，判断是核心指标（例如 acc 的 mean/maj/best @N_max）还是辅助指标，
        # 生成最终的扁平化 metric_dict，键格式如 val-core/math/acc/mean@4。
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val
        # 如果存在多轮对话样本，计算轮数的最小值、最大值和均值，加入指标字典。
        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        ====== init_workers() 的作用 ======

        创建并初始化所有 WorkerGroup：
        1. 创建 GPU 资源池（PlacementGroup）
        2. 为每个 Role 创建 WorkerGroup
        3. 初始化模型（加载 checkpoint）

        ====== WorkerGroup 创建流程 ======

        Step 1: create_resource_pool()     # 创建 GPU 资源池
        Step 2: 为每个 Role 创建 RayClassWithInitArgs  # 包装 Worker 类和配置
        Step 3: create_colocated_worker_cls()  # 合并同一资源池的多个 Role
        Step 4: RayWorkerGroup()           # 创建 WorkerGroup 实例
        Step 5: wg.spawn()                 # 创建实际的 Worker Actor
        Step 6: wg.init_model()            # 加载模型 checkpoint

        ====== RayClassWithInitArgs ======

        RayClassWithInitArgs 包装 Worker 类和初始化参数：
        - cls: Worker 类（如 ActorRolloutRefWorker）
        - config: 该 Role 的配置
        - role: Role 名称（用于命名和区分）

        ====== resource_pool_to_cls ======

        resource_pool_to_cls 存储资源池 → Role 映射：
        {
            global_pool: {
                "ActorRollout": RayClassWithInitArgs(...),
                "Critic": RayClassWithInitArgs(...),
            },
            reward_pool: {
                "RewardModel": RayClassWithInitArgs(...),
            },
        }

        同一资源池的多个 Role 会合并为一个 WorkerGroup（GPU 复用）。

        ====== create_colocated_worker_cls ======

        create_colocated_worker_cls(): 合并同一资源池的多个 Role
        - 输入：class_dict = {"ActorRollout": ..., "Critic": ...}
        - 输出：一个合并的 Worker 类，包含多个 Role 的功能

        合并好处：
        - 一个 GPU 上运行多个模型（Actor + Critic + Ref）
        - 减少 GPU 占用，提高资源利用率

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        # ====== Step 1: 创建资源池 ======
        # create_resource_pool(): 创建 PlacementGroup
        # PlacementGroup 是 Ray 的 GPU 资源调度单位
        self.resource_pool_manager.create_resource_pool()

        # ====== Step 2: 创建 resource_pool_to_cls 映射 ======
        # resource_pool_to_cls: 资源池 → Role → RayClassWithInitArgs
        # 用于后续创建 WorkerGroup
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # ====== Step 3: 创建 Actor Rollout Worker ======
        # create actor and rollout
        # actor_role: 确定 Actor 的 Role 类型
        # Role.ActorRolloutRef: 新版（包含 Ref Policy）
        # Role.ActorRollout: 旧版（不包含 Ref Policy）
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout

        if self.hybrid_engine:
            # get_resource_pool(): 获取 Actor 对应的资源池
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)

            # RayClassWithInitArgs: 包装 Worker 类和初始化参数
            # - cls: Worker 类（ActorRolloutRefWorker）
            # - config: Actor 配置
            # - role: Role 名称（用于命名）
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )

            # 存入 resource_pool_to_cls
            # 同一资源池的多个 Role 会合并为一个 WorkerGroup
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # ====== Step 4: 创建 Critic Worker ======
        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            # omega_conf_to_dataclass(): 将 OmegaConf 配置转为 dataclass
            # 方便类型检查和参数访问
            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            # 新版 Worker 实现（use_legacy_worker_impl="disable"）
            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                # TrainingWorkerConfig: 新版通用训练 Worker 配置
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg

                # FSDP 策略：配置 FSDPEngineConfig
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config

                    # 配置 token 长度限制
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                # 创建 TrainingWorkerConfig
                # TrainingWorker 可用于 Actor、Critic、Reward Model
                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",  # 指定模型类型为价值模型
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            # RayClassWithInitArgs: 包装 Critic Worker 类
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            # Note: mode is always "async" since sync mode is deprecated
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rm_resource_pool=rm_resource_pool,
        )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False)
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.

        ====== fit() 的作用 ======

        fit() 执行 PPO 训练的主循环：
        1. 加载 checkpoint（如果存在）
        2. 执行初始验证（可选）
        3. 循环执行 PPO 9 步训练流程
        4. 定期验证和保存 checkpoint

        ====== PPO 9 步训练流程 ======

        每个 batch 执行以下 9 步：

        Step 1: prepare batch          # 准备数据，添加 uid
        Step 2: generate_sequences()   # Actor 生成 response
        Step 3: compute_reward()       # 计算 reward score
        Step 4: compute_values()       # Critic 计算价值（如果需要）
        Step 5: apply_kl_penalty()     # 应用 KL 惩罚（如果启用）
        Step 6: compute_advantage()    # 计算 advantage（GAE/GRPO）
        Step 7: update_critic()        # 更新 Critic（如果需要）
        Step 8: update_actor()         # 更新 Actor（PPO clip）
        Step 9: log metrics            # 记录 metrics

        ====== Driver vs Worker ======

        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.

        Driver 进程（不占 GPU）：
        - 执行轻量计算（advantage、metrics）
        - 调用 WorkerGroup 的 RPC 方法
        - 协调数据流

        Worker 进程（占 GPU）：
        - 执行重量计算（generate、update）
        - 通过 RPC 被 Driver 调用

        The light-weight advantage computation is done on the driver process.

        ====== 数据流示意 ======

        batch_dict (from dataloader)
            ↓
        DataProto.from_single_dict(batch_dict)
            ↓
        batch.repeat(n) → 每个 prompt 复制 n 次
            ↓
        actor_rollout_wg.generate_sequences() → 生成 response
            ↓
        reward_fn.compute_reward() → 计算 reward
            ↓
        critic_wg.compute_values() → 计算 value（GAE 需要）
            ↓
        compute_advantage() → 计算 advantage（Driver 进程）
            ↓
        actor_rollout_wg.update_policy() → 更新 Actor
            ↓
        critic_wg.update_value() → 更新 Critic
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        # ====== Step 0: 初始化 logger ======
        # Tracking: 初始化 wandb/tensorboard logger
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # ====== Step 1: 加载 checkpoint ======
        # load checkpoint before doing anything
        # _load_checkpoint(): 从 checkpoint 恢复训练状态
        # - 恢复 global_steps、dataloader state、optimizer state
        self._load_checkpoint()

        # 计算当前 epoch（用于恢复训练）
        current_epoch = self.global_steps // len(self.train_dataloader)

        # ====== Step 2: 执行初始验证 ======
        # perform validation before training
        # currently, we only support validation using the reward_function.
        # val_before_train: 训练前先验证，记录初始性能基线
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)

            # val_only: 只验证不训练（用于检查 reward function）
            if self.config.trainer.get("val_only", False):
                return

        # ====== Step 3: 配置 rollout skip（可选）======
        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # ====== Step 4: 初始化进度条 ======
        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        # ====== Step 5: 配置 profiling ======
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # ====== Step 6: 开始训练循环 ======
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                # async_rollout: 完成之前的异步 rollout 调用
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)

                metrics = {}
                timing_raw = {}

                # ====== Step 6.1: 开始 profiling ======
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                # ====== Step 6.2: 准备 batch ======
                # batch_dict → DataProto
                # DataProto.from_single_dict(): 将字典转为 DataProto
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # 设置生成温度参数
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                # uid: 每个 prompt 的唯一 ID
                # 用于分组：同一 prompt 的多个 response 共享 uid
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # ====== Step 6.3: 分离生成数据 ======
                # _get_gen_batch(): 从完整 batch 分离出生成需要的部分
                # - 保留 uid、reward_model 等元信息在原 batch
                # - gen_batch 只包含 input_ids 等生成需要的字段
                gen_batch = self._get_gen_batch(batch)

                # ====== Step 6.4: 复制 prompt（repeat）======
                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps

                # repeat(): 每个 prompt 复制 n 次
                # n: 每个 prompt 生成多少个 response
                # interleave=True: [p1, p1, p1, p1, p2, p2, p2, p2, ...]
                # 用于 GRPO（需要同一 prompt 的多个 response）
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                # ====== Step 6.5: 开始一个训练步 ======
                with marked_timer("step", timing_raw):
                    # ====== Step 7: Generate Sequences ======
                    # generate a batch
                    # generate_sequences(): Actor 生成 response
                    # - 输入：gen_batch（只有 prompt）
                    # - 输出：gen_batch_output（prompt + response）
                    # - 添加：responses, input_ids, attention_mask, old_log_probs
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            # 同步模式：直接调用 WorkerGroup
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            # 异步模式：通过 async_rollout_manager
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        # 记录生成时间
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                    # 优势估计器（Advantage Estimator）是负责计算“优势函数（Advantage Function）”的核心模块。
                    # 它用来量化一个动作比“平均水平”好多少，是指导模型学习和更新的关键“导航仪”
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            # 深拷贝一份生成输入，强制 do_sample=False，即用贪婪解码，得到确定性最强的回答作为 baseline。
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            # 输出 gen_batch_output 是一个 DataProto，
                            # 包含 responses、old_log_probs（如果生成时计算了）、attention_mask 等。
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            # 如果需要神经网络 RM 打分，且尚未算好，就调用 Reward Worker 计算 RM 分数，合并进 batch。
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            # 调用统一奖励接口，计算 baseline 回答的总奖励（sum_reward=True 把多维分数压缩成一个标量）。
                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )
                            # 从 batch 中移除只为算 baseline 而临时加入的张量（baseline responses、rm_scores），
                            # 只保留 reward_baselines 这个标量，作为后续 REMAX 优势计算的基线。
                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            # 手动删除中间变量，释放显存。
                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # ====== Step 8: 合并 prompt 和 response ======
                    # repeat to align with repeated responses in rollout
                    # batch（原始 prompt，未 repeat）也需要 repeat 以匹配 gen_batch_output
                    # 把原始 batch（通常是一个 prompt 一份）复制 n 份，与生成时每个 prompt 生成 n 个 responses 对齐。
                    # interleave=True 保证同一个 prompt 的 n 个副本连续，便于后续分组计算优势。
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                    # union(): 合并 batch 和 gen_batch_output
                    # - batch 包含 uid、reward_model 等元信息
                    # - gen_batch_output 包含 responses、old_log_probs 等
                    # 把生成结果（responses、old_log_probs 等）与原始 batch（uid、reward_model 等元信息）合并。
                    # 现在 batch 是完整的“prompt + response + 元信息”数据包。
                    batch = batch.union(gen_batch_output)

                    # 计算 response_mask（如果不存在）
                    # 若还没生成 response 的 mask，就基于 prompt 长度计算一个，标记哪些 token 属于生成部分（不包含 prompt），用于后续损失聚合。
                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # ====== Step 9: 平衡 batch（可选）======
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # balance_batch: 按序列长度平衡负载（长序列和短序列均匀分布）
                    # 如果开启 balance_batch，会按序列长度重新排列样本，让各个数据并行 GPU 分到的有效 token 数尽量均匀，
                    # 提升训练效率。可能改变样本顺序，但 advantage 计算基于 uid 分组，不受影响。
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    # 记录每个样本的有效 token 数
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # ====== Step 10: Compute Reward ======
                    # compute reward
                    # reward 来源：
                    # 1. Reward Model Worker（use_rm=True）
                    # 2. Reward Function（reward_fn）
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        # 如果启用 Reward Model Worker，计算 RM score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                # 同步模式：直接调用 rm_wg
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                # 异步模式：通过 reward_loop_manager
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Compute or extract reward for training
                        # _compute_or_extract_reward(): 计算 reward score
                        # - 使用 reward_fn（规则函数）
                        # - 或者提取已有的 token_level_scores
                        # 如果配置为异步执行奖励函数（规则型），就在 Ray 上启动一个异步任务，
                        # 返回 future_reward 对象，后续再用 ray.get 取结果。
                        # 否则同步调用 _compute_or_extract_reward，当场拿到 reward_tensor 和额外信息。
                        if self.config.reward_model.launch_reward_fn_async:
                            # 异步计算 reward（不阻塞训练）
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            # 同步计算 reward
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, return_dict=False
                            )

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates

                    # ====== Step 11: 计算 old_log_prob ======
                    # old_log_prob: PPO 算法需要的"旧策略" log probability
                    # 用于计算 ratio = π(a|s) / π_old(a|s)
                    # 检查是否配置了绕过重新计算 log prob 的旁路模式。
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    # 如果启用，直接使用 rollout 时 vLLM 等引擎计算好的 rollout_log_probs 作为 old_log_probs，省去一次模型推理。
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        # Bypass mode: 直接使用 rollout 时计算的 log_probs
                        # 适合：rollout 和 training 使用相同的策略
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        # Decoupled mode: 重新计算 old_log_prob
                        # 适合：rollout 使用 vLLM（推理引擎），training 使用训练引擎
                        # 需要重新计算以确保一致性
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            # _compute_old_log_prob(): 计算当前策略的 log probability
                            # - 输入：batch（包含 input_ids, responses）
                            # - 输出：old_log_prob（包含 old_log_probs, entropys）
                            # 否则，调用 _compute_old_log_prob 在训练引擎上重新计算当前策略下生成 token 的对数概率。
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)

                            # 计算 entropy（用于记录 metrics）
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor

                            # agg_loss(): 聚合 entropy
                            # loss_agg_mode: token-mean 或 sequence-mean
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )

                            # 记录 entropy 和 MFU metrics
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)

                            # 移除 entropys（不需要存入 batch）
                            old_log_prob.batch.pop("entropys")

                            # union(): 将 old_log_prob 合入 batch
                            batch = batch.union(old_log_prob)

                            # 计算 rollout_log_probs 和 old_log_probs 的差异（用于调试）
                            # 如果有 rollout 引擎留下的 log prob，计算它与刚算的 old_log_probs 的差异（K-L 散度等），辅助调试。
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    # 验证 old_log_probs 存在
                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    # ====== Step 12: Compute Ref Log Prob ======
                    # 如果使用参考策略（Reference Model），调用 _compute_ref_log_prob 得到 ref_log_prob，用于计算 KL 惩罚。
                    if self.use_reference_policy:
                        # compute reference log_prob
                        # _compute_ref_log_prob(): 计算参考策略的 log probability
                        # 用于 KL 惩罚：KL = log_prob_current - log_prob_ref
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # ====== Step 13: Compute Values ======
                    # compute values
                    # _compute_values(): Critic 计算每个状态的价值 V(s)
                    # 用于 GAE advantage 计算
                    # 只有 PPO 等需要 Critic 的算法才执行。Critic 估计每个 token 的状态价值 V(s)，用于 GAE 优势计算。
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    # ====== Step 14: Compute Advantage ======
                    # compute advantage（在 Driver 进程计算）
                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]

                        # 异步 reward：等待结果
                        # 如果奖励函数是异步启动的，现在才真正阻塞获取结果。
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)

                        # token_level_scores: 每个 token 的 reward score
                        batch.batch["token_level_scores"] = reward_tensor

                        # reward_extra_infos_dict: 额外信息（如 ground_truth）
                        # 把额外奖励信息（如准确率）转成 numpy 数组，存入 non_tensor_batch。
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        # apply_kl_penalty(): 应用 KL 惩罚（如果启用）
                        # reward = task_reward - β * KL(π || π_ref)
                        # 如果配置了将 KL 惩罚直接加到奖励里，就调用 apply_kl_penalty 修改奖励；否则直接将 token_level_scores 复制为 token_level_rewards。
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            # 如果不启用 KL in reward，直接使用 token_level_scores
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        # Rollout correction: 修正 rollout 和 training 使用不同引擎的问题
                        # 在解耦模式下（训练引擎重新计算了 old_log_probs，且 rollout 引擎保留了 rollout_log_probs），
                        # 可以计算重要性采样修正、拒绝采样等，解决训练和推理引擎分布差异问题。
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        # compute_advantage(): 计算 advantage（在 Driver 进程）
                        # - GAE: A = Σ (γλ)^l * (r_l + γV(s_{l+1}) - V(s_l))
                        # - GRPO: A = (r - mean) / std（同一 prompt 的 n 个 response）
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # ====== Step 15: Update Critic ======
                    # update critic
                    # _update_critic(): 更新 Critic 参数
                    # Loss: (V - return)^2
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        # reduce_metrics(): 聚合所有 Worker 的 metrics
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # ====== Step 16: Update Actor ======
                    # implement critic warmup
                    # critic_warmup: 先更新 Critic 几步，让 Critic 更稳定
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        # _update_actor(): 更新 Actor 参数
                        # Loss: PPO clip loss = -min(ratio * A, clip(ratio) * A)
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                # ESI（Elastic Server Instance）：云上弹性服务器实例，通常有最大运行时长的限制。
                # should_save_ckpt_esi：根据历史最长 step 耗时和预留冗余时间，判断当前 ESI 是否快要被系统回收。
                # 如果快到期，返回 True，强制触发后续保存。
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                # 保存条件：save_freq > 0 且满足下列之一：
                # 最后一步
                # 步数是 save_freq 的倍数（如每 100 步保存）
                # ESI 即将过期（强制保存）
                # 执行：绿色计时，调用 self._save_checkpoint() 持久化模型参数、优化器状态等。
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()
                # 确定下一步是否需要 profile：如果配置了 global_profiler.steps（一组步数），检查 global_steps + 1 是否在其中。
                # 停止条件：
                # 连续 profile 模式（profile_continuous_steps=True）：只有当前步需要 profile 且下一步不需要时才停止，确保连续步之间不中断采样。
                # 非连续模式：只要当前步在 profile，就立刻停止。
                # 状态推移：把 curr_step_profile 变成 prev_step_profile，next_step_profile 变成新的 curr_step_profile，为下一步 prep。
                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile
                # 取出当前步总耗时（"step" 计时器的值），更新历史最大步耗时（用于 ESI 估算）。
                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                # 添加当前全局步数和 epoch 信息到 metrics。
                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                # compute_data_metrics：统计 batch 数据相关的指标，如 prompt 长度、response 长度、有效 token 数、奖励均值等。
                # compute_timing_metrics：从 timing_raw 提取各阶段耗时（gen, reward, adv, update 等），拼成 perf/timing/xxx 格式。
                # compute_throughout_metrics：结合 GPU 数量和耗时，计算每秒处理的 token 数、sample 数等吞吐指标。
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation
                # 如果使用了课程学习采样器（根据训练进度动态调整数据难度），就把当前 batch 的信息传给采样器，供其调整后续数据分布。
                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)
                # 调用统一日志接口 logger.log，将本步所有 metrics 写入 wandb / TensorBoard / MLflow 等。
                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                # 进度条走一格，全局步数加1
                progress_bar.update(1)
                self.global_steps += 1
                # 如果启用了 PyTorch 内存分析器，就在这一步的更新后保存一份显存快照，用于排查显存泄漏或峰值
                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )
                # 异步调用收尾：如果 WorkerGroup 有异步执行的清理函数，就阻塞等待它们全部完成，确保所有任务干净结束。
                # 打印最后的验证指标：把 last_val_metrics 输出到控制台，方便训练结束瞬间查看。
                # 关闭进度条，然后 return：结束整个训练循环。
                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
                # 如果训练数据集对象实现了 on_batch_end 方法（如动态数据增强或日志记录），
                # 就在每个 batch 处理完后调用它。此功能标记为实验性，可能后续调整。
                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
