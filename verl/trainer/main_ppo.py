# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
====== main_ppo.py 整体流程 ======

这是 PPO 训练的入口文件，负责：
1. 解析配置（通过 Hydra）
2. 初始化 Ray 集群
3. 创建 TaskRunner Actor（在独立进程中运行）
4. TaskRunner.run() 执行完整训练流程

====== 文件结构 ======

main(config)           # 入口函数，Hydra 装饰器
run_ppo(config)        # 初始化 Ray，创建 TaskRunner
TaskRunner.run(config) # 主训练流程（在 Ray Actor 中执行）

====== TaskRunner.run() 7 阶段流程 ======

Step 1: add_actor_rollout_worker()  # 创建 Actor WorkerGroup
Step 2: add_critic_worker()         # 创建 Critic WorkerGroup
Step 3: add_reward_model_worker()   # 创建 RM WorkerGroup（可选）
Step 4: add_ref_policy_worker()     # 创建 Ref WorkerGroup（可选）
Step 5: init_resource_pool_mgr()    # 创建 GPU 资源池
Step 6: create_rl_dataset_sampler() # 创建数据集和采样器
Step 7: RayPPOTrainer.fit()         # 执行训练循环

====== 数据流示意 ======

配置文件 (ppo_trainer.yaml)
    ↓
main() → Hydra 解析配置
    ↓
run_ppo() → 初始化 Ray 集群
    ↓
TaskRunner.remote() → 创建 Actor
    ↓
TaskRunner.run() → 7 阶段初始化
    ↓
RayPPOTrainer.fit() → PPO 训练循环
    ↓
logger.log() → 记录 metrics

Note that we don't combine the main with ray_trainer as ray_trainer is used by other mpain.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device, is_cuda_available
from verl.utils.import_utils import load_extern_object


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)

    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            # Add runtime environment variables for transfer queue
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)  # please make sure main_task is not scheduled on head

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    ====== TaskRunner 是什么？======

    TaskRunner 是一个 Ray Actor，运行在独立的进程中。
    它负责协调整个训练流程：

    1. 创建各种 WorkerGroup（Actor, Critic, RM, Ref）
    2. 管理 GPU 资源池
    3. 创建数据集和采样器
    4. 启动 RayPPOTrainer 进行训练

    ====== 为什么需要 TaskRunner？======

    Ray 的分布式训练需要一个"Driver"进程来协调：
    - Driver 不占用 GPU（只做调度）
    - Worker 占用 GPU（做实际计算）
    - TaskRunner 就是 Driver

    通过 ray.remote()，TaskRunner 成为 Ray Actor：
    - 可以被远程调用
    - 运行在独立进程
    - 不阻塞主进程

    ====== TaskRunner vs Worker ======

    | 角色 | 进程类型 | GPU | 作用 |
    |------|----------|-----|------|
    | TaskRunner | Driver Actor | 0 | 协调、调度 |
    | Worker | Worker Actor | 1 | 模型计算 |

    ====== 调用链 ======

    main()
      ↓
    run_ppo()
      ↓
    runner = TaskRunner.remote()  # 创建 Actor
      ↓
    ray.get(runner.run.remote(config))  # RPC 调用，等待完成

    Attributes:
        role_worker_mapping: Role → Worker 类的映射
            例如：{Role.ActorRollout: ActorRolloutRefWorker}
        mapping: Role → resource_pool 的映射
            例如：{Role.ActorRollout: "global_pool"}
    """

    def __init__(self):
        # role_worker_mapping: 存储 Role → Worker 类的映射
        # 用于后续创建 WorkerGroup
        self.role_worker_mapping = {}
        # mapping: 存储 Role → 资源池名称的映射
        # 决定每个 Role 使用哪个 GPU 资源池
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy.

        ====== 这个方法做什么？======

        根据配置选择合适的 Actor Worker 类，并注册到 role_worker_mapping。

        ====== Actor Worker 的作用 ======

        Actor Worker 负责：
        1. 用当前策略模型生成 response（Rollout）
        2. 计算旧策略的 log probability（用于 PPO clip）
        3. 如果启用 KL，同时作为 Reference Policy（Ref）

        ====== Worker 选择逻辑 ======

        | 配置 | 选择的 Worker 类 |
        |------|------------------|
        | use_legacy_worker_impl="disable" | ActorRolloutRefWorker（新版） |
        | strategy="fsdp"/"fsdp2" | AsyncActorRolloutRefWorker |
        | strategy="megatron" | AsyncActorRolloutRefWorker |

        ====== Role 映射示例 ======

        role_worker_mapping 最终结果：
        {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),  # Actor+Rollout
            Role.ActorRolloutRef: ray.remote(ActorRolloutRefWorker),  # Actor+Rollout+Ref（新版）
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker),  # Reference Policy（旧版）
        }

        ====== 为什么用 ray.remote()？======

        ray.remote(WorkerClass) 将 Worker 类变成 Ray Actor：
        - 可以远程调用（在 GPU 进程中运行）
        - 返回 ActorClass（不是实例，是可创建 Actor 的类）
        - 后续通过 WorkerGroup 创建多个实例
        """
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        # use_legacy_worker_impl: 控制使用新版还是旧版 Worker 实现
        # "auto"/"enable": 旧版实现（分离的 Actor/Rollout/Ref Worker）
        # "disable": 新版实现（统一的 ActorRolloutRefWorker）
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # ====== 新版 Model Engine（统一 Worker）======
        # use_legacy_worker_impl == "disable" 时使用新版
        # 新版特点：Actor、Rollout、Ref Policy 统一在一个 Worker 中
        if use_legacy_worker_impl == "disable":
            from verl.workers.engine_workers import ActorRolloutRefWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

            # 新版 Worker 的 Role 选择逻辑：
            # - 如果启用 KL（in reward 或 as loss），需要 Ref Policy，使用 ActorRolloutRef
            # - 如果不启用 KL，只需要 Actor+Rollout，使用 ActorRollout
            # NOTE: In new model engine, ref policy and actor rollout are in same ActorRolloutRefWorker,
            # while in legacy model engine, ref policy is in a separate ActorRolloutRefWorker.
            if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
                role = Role.ActorRolloutRef  # 包含 Actor + Rollout + Reference Policy
            else:
                role = Role.ActorRollout  # 只包含 Actor + Rollout

            # ray.remote(actor_rollout_cls): 将 Worker 类变成 Ray ActorClass
            # 后续 WorkerGroup 会用这个类创建多个 Worker 实例
            self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)

            # mapping[role] = "global_pool": 表示这个 Role 使用名为 "global_pool" 的资源池
            # 资源池在 init_resource_pool_mgr() 中创建
            self.mapping[role] = "global_pool"
            return actor_rollout_cls, ray_worker_group_cls

        # ====== 旧版 Model Engine（分离 Worker）======
        # use_legacy_worker_impl == "auto" 或 "enable" 时使用旧版
        # 旧版特点：Actor/Rollout 使用 AsyncActorRolloutRefWorker，Ref Policy 是单独的 Worker

        # Note: sync mode validation is now handled in RolloutConfig.__post_init__
        # Always use async worker since sync mode is deprecated and rejected

        # strategy: 决定使用哪种分布式训练框架
        # "fsdp"/"fsdp2": 使用 PyTorch FSDP（Fully Sharded Data Parallel）
        # FSDP 将模型参数分片到多个 GPU，适合大模型训练
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        # "megatron": 使用 Megatron-LM 分布式策略
        # Megatron 使用 Tensor Parallel + Pipeline Parallel，适合超大模型
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        # 旧版只注册 ActorRollout Role，Ref Policy 在 add_ref_policy_worker() 中单独注册
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping.

        ====== Critic Worker 的作用 ======

        Critic（价值函数）用于：
        1. 估计每个状态的价值 V(s)
        2. 计算 Advantage（GAE 需要 Critic）
        3. 不需要 Critic 的算法：GRPO（用 group mean/std 计算 advantage）

        ====== Worker 选择逻辑 ======

        | 配置 | 选择的 Worker 类 |
        |------|------------------|
        | use_legacy_worker_impl="disable" | TrainingWorker（新版通用） |
        | strategy="fsdp"/"fsdp2" + legacy | CriticWorker（FSDP 专用） |
        | strategy="megatron" | CriticWorker（Megatron 专用） |

        ====== 新版 TrainingWorker 设计 ======

        新版用 TrainingWorker 作为通用训练 Worker：
        - Actor、Critic、Reward Model 都可以用 TrainingWorker
        - 减少代码重复，统一训练逻辑
        """
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        # strategy 选择：FSDP 或 Megatron
        # FSDP: 全分片数据并行，每个 GPU 只保存部分参数
        # Megatron: Tensor Parallel + Pipeline Parallel，适合超大模型
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            # 旧版使用专门的 CriticWorker
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker

            # 新版使用通用 TrainingWorker
            elif use_legacy_worker_impl == "disable":
                # we don't need to specialize critic worker. Just use TrainingWorker
                # TrainingWorker 是通用训练 Worker，可用于 Actor/Critic/RM
                from verl.workers.engine_workers import TrainingWorker

                CriticWorker = TrainingWorker
                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        elif config.critic.strategy == "megatron":
            # TODO: switch this to TrainingWorker as well
            # Megatron 目前仍使用专门的 CriticWorker
            from verl.workers.megatron_workers import CriticWorker

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        # 注册 Critic Worker 到 role_worker_mapping
        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

        # Critic 使用 global_pool 资源池（与 Actor 共享 GPU，通过 max_colocate_count 复用）
        self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager.

        ====== ResourcePoolManager 的作用 ======

        管理所有 GPU 资源池：
        1. 定义每个资源池的 GPU 配置（每节点多少 GPU）
        2. 创建 PlacementGroup（Ray 的资源调度单位）
        3. 将 Role 映射到对应的资源池

        ====== 资源池配置示例 ======

        假设配置：
        - trainer.nnodes = 2（两个节点）
        - trainer.n_gpus_per_node = 8（每节点 8 GPU）
        - reward_model.enable_resource_pool = True
        - reward_model.nnodes = 1, reward_model.n_gpus_per_node = 4

        则 resource_pool_spec 结果：
        {
            "global_pool": [8, 8],  # 两个节点各 8 GPU，共 16 GPU
            "reward_pool": [4],     # 一个节点 4 GPU，单独资源池
        }

        ====== Role → 资源池映射 ======

        | Role | 资源池 | 说明 |
        |------|--------|------|
        | ActorRollout | global_pool | Actor 用主资源池 |
        | Critic | global_pool | Critic 用主资源池 |
        | RefPolicy | global_pool | Ref 用主资源池 |
        | RewardModel | reward_pool（如果启用） | RM 用独立资源池 |

        ====== 为什么 Reward Model 用独立资源池？======

        Reward Model 计算量大（需要推理整个 response）：
        - 独立资源池避免与 Actor/Critic 争抢 GPU
        - 可以异步计算 reward（不阻塞训练）
        - 提高整体吞吐量
        """
        # global_pool: 主资源池，用于 Actor、Critic、Ref Policy
        global_pool_id = "global_pool"

        # resource_pool_spec: 资源池配置字典
        # 格式：{pool_name: [n_gpus_per_node] * nnodes}
        # 例如 [8] * 2 = [8, 8]，表示两个节点各 8 GPU
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # TODO Here you can use the new registration method to support dynamic registration of roles

        # reward_pool: Reward Model 专用资源池（可选）
        # 启用后，RM 不会与 Actor/Critic 共享 GPU，避免资源竞争
        if config.reward_model.enable_resource_pool:
            # 验证 Reward Model 资源池配置
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")

            # 创建 reward_pool 配置
            # 例如 nnodes=1, n_gpus_per_node=4 → [4]
            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        # 创建 ResourcePoolManager 实例
        # 它会根据 resource_pool_spec 创建 PlacementGroup
        # 并根据 self.mapping 将 Role 分配到对应资源池
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled.

        ====== Reward Model Worker 的作用 ======

        Reward Model 用于计算 response 的 reward score：
        1. 基于模型的 RM：用神经网络评估 response 质量
        2. 规则 RM：用规则函数计算 reward（如代码执行结果）
        3. 混合 RM：模型 + 规则组合

        ====== Reward Model vs Reward Function ======

        | 类型 | 配置 | 计算 |
        |------|------|------|
        | Reward Function | config.reward_fn | Python 函数（规则） |
        | Reward Model Worker | config.reward_model.enable | GPU 模型推理 |

        ====== 资源池选择 ======

        | enable_resource_pool | 资源池 | 说明 |
        |-----------------------|--------|------|
        | True | reward_pool | 独立 GPU，避免竞争 |
        | False | global_pool | 与 Actor/Critic 共享 GPU |

        ====== 为什么 Reward Model 需要 GPU？======

        Neural Reward Model（如 LLM-as-a-Judge）：
        - 输入：prompt + response
        - 输出：reward score（0-1 或 -1-1）
        - 需要 GPU 进行推理
        """
        from verl.trainer.ppo.ray_trainer import Role

        # config.reward_model.enable: 是否启用 Neural Reward Model
        # 如果禁用，只使用 Reward Function（规则计算）
        if config.reward_model.enable:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

            # Worker 选择：新版/旧版 + FSDP/Megatron
            if use_legacy_worker_impl in ["auto", "enable", "disable"]:
                if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                    from verl.workers.fsdp_workers import RewardModelWorker
                elif config.reward_model.strategy == "megatron":
                    from verl.workers.megatron_workers import RewardModelWorker
                else:
                    raise NotImplementedError

            # elif use_legacy_worker_impl == "disable":
            #     from verl.workers.engine_workers import RewardModelWorker
            #
            #     print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            # 注册 Reward Model Worker
            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)

            # 资源池选择：
            # - enable_resource_pool=True: 使用独立 reward_pool
            # - enable_resource_pool=False: 共享 global_pool
            if config.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used.

        ====== Reference Policy 的作用 ======

        Reference Policy（冻结的初始策略）用于计算 KL 约束：
        1. KL in Reward：reward = task_reward - β * KL(π || π_ref)
        2. KL Loss：loss = ppo_loss + β * KL(π || π_ref)

        ====== 为什么需要 KL 约束？======

        PPO 可能过度优化，导致：
        - 策略偏离原始模型太多
        - 生成内容质量下降（如重复、乱码）
        - KL 约束防止过度偏离

        ====== 新版 vs 旧版架构 ======

        | 版本 | Ref Policy 存放位置 |
        |------|---------------------|
        | 新版（use_legacy_worker_impl="disable"） | ActorRolloutRefWorker 内部 |
        | 旧版 | 独立的 RefPolicy Worker |

        ====== 什么时候启用 Ref Policy？======

        条件：use_kl_in_reward 或 use_kl_loss 为 True
        """
        from verl.trainer.ppo.ray_trainer import Role

        # Ref policy has been fused into ActorRolloutRefWorker in new model engine,
        # we don't need to add a separate ref policy worker group.
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # 新版 Model Engine：Ref Policy 已融合到 ActorRolloutRefWorker
        # 不需要单独注册 RefPolicy Worker
        if use_legacy_worker_impl == "disable":
            return

        # 旧版 Model Engine：需要单独的 RefPolicy Worker
        # 条件：启用 KL in reward 或 KL loss
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            # ref_policy_cls 是 add_actor_rollout_worker 返回的 Worker 类
            # Ref Policy 与 Actor 使用相同的 Worker 类，但 Role 不同
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)

            # Ref Policy 使用 global_pool（与 Actor 共享 GPU）
            self.mapping[Role.RefPolicy] = "global_pool"

    def run(self, config):
        """Execute the main PPO training workflow.

        ====== run() 是什么？======

        run() 是 PPO 训练的核心入口，执行完整的 7 阶段初始化流程。
        它在 TaskRunner Actor 中运行（独立进程，不占用 GPU）。

        ====== 7 阶段流程 ======

        Step 1: add_actor_rollout_worker()  # 创建 Actor WorkerGroup
        Step 2: add_critic_worker()         # 创建 Critic WorkerGroup
        Step 3: add_reward_model_worker()   # 创建 RM WorkerGroup（可选）
        Step 4: add_ref_policy_worker()     # 创建 Ref WorkerGroup（可选）
        Step 5: init_resource_pool_mgr()    # 创建 GPU 资源池
        Step 6: create_rl_dataset_sampler() # 创建数据集和采样器
        Step 7: RayPPOTrainer.fit()         # 执行训练循环

        ====== 数据流示意 ======

        配置文件 → Hydra → config dict
            ↓
        TaskRunner.run(config)
            ↓
        创建 Workers（Step 1-4）
            ↓
        创建 ResourcePool（Step 5）
            ↓
        创建 Dataset（Step 6）
            ↓
        RayPPOTrainer.fit() → 9 步训练循环

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # ====== Step 0: 配置解析 ======
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        # 打印当前进程信息，用于调试分布式问题
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        # 打印完整配置，resolve=True 会解析所有引用（如 ${actor.path}）
        pprint(OmegaConf.to_container(config, resolve=True))

        # resolve(): 解析配置中的所有插值引用
        # 例如 ${actor_rollout_ref.model.path} → "/path/to/model"
        OmegaConf.resolve(config)

        # ====== Step 1-4: 创建 Worker 映射 ======
        # 这些方法不创建实际 Worker，只注册 Role → WorkerClass 映射
        # 实际 Worker 在 RayPPOTrainer.init_workers() 中创建

        # Step 1: 注册 Actor Rollout Worker
        # 返回 actor_rollout_cls（用于 Ref Policy）和 ray_worker_group_cls（用于 Trainer）
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)

        # Step 2: 注册 Critic Worker
        self.add_critic_worker(config)

        # Step 3: 注册 Reward Model Worker（可选）
        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        self.add_reward_model_worker(config)

        # Step 4: 注册 Reference Policy Worker（可选）
        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # ====== Step 5: 验证配置 ======
        # validate config
        # 检查配置一致性：
        # - 如果有 Ref Policy，必须启用 KL
        # - 如果用 GAE，必须有 Critic
        # - 如果用 GRPO，不需要 Critic
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        # ====== Step 6: 加载模型和 Tokenizer ======
        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        # copy_to_local(): 从远程存储（HDFS/S3）下载模型到本地
        # use_shm=True: 使用共享内存，多进程共享模型权重
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        # trust_remote_code: 是否执行模型仓库中的自定义代码
        # Qwen、ChatGLM 等模型需要 trust_remote_code=True
        trust_remote_code = config.data.get("trust_remote_code", False)

        # hf_tokenizer(): 加载 HuggingFace tokenizer
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # processor: 多模态模型需要 processor（如图像预处理）
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # ====== Step 7: 加载 Reward Manager ======
        # Load the reward manager for training and validation.
        # reward_fn: 训练时的 reward 计算函数
        # num_examine=0: 训练时不记录样本详情（避免内存占用）
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )

        # val_reward_fn: 验证时的 reward 计算函数
        # num_examine=1: 验证时记录样本详情（用于 wandb 可视化）
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        # ====== Step 8: 创建资源池 ======
        # resource_pool_manager: 管理 GPU 资源分配
        # 创建 PlacementGroup，将 Role 分配到对应资源池
        resource_pool_manager = self.init_resource_pool_mgr(config)

        # ====== Step 9: 创建数据集 ======
        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        # create_rl_dataset(): 创建 RL 训练数据集
        # 数据格式：{"prompt": ..., "ground_truth": ..., "reward_model": ...}
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )

        # train_sampler: 数据采样器
        # 支持随机采样、顺序采样、课程学习采样
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # ====== Step 10: 创建 RayPPOTrainer ======
        # Initialize the PPO trainer.
        # RayPPOTrainer 是 PPO 训练的主控制器：
        # - 管理 WorkerGroup（Actor, Critic, RM, Ref）
        # - 执行 9 步训练循环（generate → reward → advantage → update）
        # - 处理 checkpoint 和 validation
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,  # Role → WorkerClass 映射
            resource_pool_manager=resource_pool_manager,  # GPU 资源池管理
            ray_worker_group_cls=ray_worker_group_cls,  # WorkerGroup 类
            reward_fn=reward_fn,  # 训练 reward 函数
            val_reward_fn=val_reward_fn,  # 验证 reward 函数
            train_dataset=train_dataset,  # 训练数据集
            val_dataset=val_dataset,  # 验证数据集
            collate_fn=collate_fn,  # 数据 collate 函数
            train_sampler=train_sampler,  # 训练采样器
        )

        # ====== Step 11: 初始化 Workers ======
        # Initialize the workers of the trainer.
        # init_workers(): 创建实际的 WorkerGroup
        # 根据 role_worker_mapping 和 resource_pool_manager 创建 Worker
        trainer.init_workers()

        # ====== Step 12: 开始训练 ======
        # Start the training process.
        # fit(): 执行 PPO 训练循环
        # 包含：generate → compute_reward → compute_advantage → update_actor → update_critic
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """Create a dataset.

    ====== 数据集格式 ======

    RL 数据集通常包含：
    - prompt: 输入问题/指令
    - ground_truth: 正确答案（用于 reward 计算）
    - reward_model: reward 计算配置（如函数名、参数）

    ====== 数据流示例 ======

    JSON 数据文件：
    [
        {"prompt": "请计算 1+1", "ground_truth": "2"},
        {"prompt": "请写一个函数", "ground_truth": "def func(): ..."},
    ]

    → dataset_cls 加载并预处理
    → RLDataset:
        - data: [{'prompt': ..., 'ground_truth': ...}, ...]
        - tokenizer: 将文本转为 token ids
    → dataloader:
        - batch_size=16
        - collate_fn: 将多个样本合并为 DataProto

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """

    from verl.utils.dataset.rl_dataset import get_dataset_class

    # Get the dataset class
    # get_dataset_class(): 根据 data_config.dataset_type 选择数据集类
    # 例如："math" → MathDataset, "code" → CodeDataset
    dataset_cls = get_dataset_class(data_config)

    # Instantiate the dataset using the determined dataset class
    # dataset_cls(): 创建数据集实例
    # - data_files: 数据文件路径
    # - tokenizer: 用于文本编码
    # - processor: 多模态处理器
    # - max_samples: 限制样本数量（用于调试）
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    ====== Sampler 的作用 ======

    Sampler 决定数据集的遍历顺序：
    1. RandomSampler: 随机顺序（训练默认）
    2. SequentialSampler: 顺序遍历（验证默认）
    3. CurriculumSampler: 课程学习（从简单到难）

    ====== 采样器选择逻辑 ======

    | 条件 | 采样器 |
    |------|--------|
    | sampler.class_path 存在 | 课程学习采样器 |
    | shuffle=True | 随机采样器 |
    | shuffle=False | 顺序采样器 |

    ====== 课程学习采样器 ======

    课程学习：从简单样本开始，逐渐增加难度
    - 需要自定义采样器类（class_path, class_name）
    - num_workers 必须为 0（避免数据缓存）

    ====== 随机采样器的 seed ======

    设置 seed 保证可恢复性：
    - checkpoint 后恢复训练时，采样顺序一致
    - 便于实验复现

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import SequentialSampler

    # torch.utils.data.RandomSampler could not recover properly
    # 使用 torchdata 的 RandomSampler，支持状态保存/恢复
    from torchdata.stateful_dataloader.sampler import RandomSampler

    # ====== 课程学习采样器 ======
    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        # load_extern_object(): 从外部路径加载采样器类
        curriculum_class = load_extern_object(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )

        # curriculum_class(): 创建课程学习采样器实例
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )

        # 验证采样器类型
        assert isinstance(sampler, AbstractSampler)

        # 课程学习需要 num_workers=0
        # 原因：如果 num_workers > 0，dataloader 会缓存数据
        # 课程学习采样器无法动态调整顺序
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )

    # ====== 随机采样器 ======
    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    elif data_config.shuffle:
        # train_dataloader_generator: 用于控制随机种子
        train_dataloader_generator = torch.Generator()

        # seed: 设置随机种子，保证可恢复性
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)

        # RandomSampler: 随机遍历数据集
        # generator: 控制随机顺序的生成器
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)

    # ====== 顺序采样器 ======
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        # SequentialSampler: 按索引顺序遍历（0, 1, 2, ...）
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
