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
====== main_dapo.py 整体架构 ======

这是 DAPO 训练的入口文件，与 main_ppo.py 结构类似。

====== main_dapo.py vs main_ppo.py ======

| 差异点 | main_ppo.py | main_dapo.py |
|--------|-------------|--------------|
| 配置文件 | ppo_trainer.yaml | dapo_trainer.yaml |
| Trainer 类 | RayPPOTrainer | RayDAPOTrainer |
| 训练循环 | PPO 9 步固定循环 | DAPO 动态采样循环 |
| reward_fn | 标准 reward | DAPO reward（含动态采样逻辑） |

====== DAPO 配置差异 ======

dapo_trainer.yaml 关键配置：

algorithm:
  adv_estimator: grpo  # DAPO 使用 GRPO（不需要 Critic）
  filter_groups:
    enable: true       # 启用动态过滤
    metric: seq_reward # 过滤指标
    max_num_gen_batches: 10  # 最大生成次数
  clip_ratio_low: 0.2  # 不对称裁剪（负样本）
  clip_ratio_high: 2.0 # 不对称裁剪（正样本）

actor_rollout_ref:
  rollout:
    n: 8  # 每个 prompt 生成 8 个 response（GRPO 需要）

====== TaskRunner.run() 流程 ======

1. 解析配置
2. 加载 tokenizer
3. 创建 Worker 类映射
4. 创建资源池
5. 创建数据集
6. 创建 RayDAPOTrainer
7. RayDAPOTrainer.fit() → DAPO 动态采样训练

Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import auto_set_device, is_cuda_available

from .dapo_ray_trainer import RayDAPOTrainer


@hydra.main(config_path="config", config_name="dapo_trainer", version_base=None)
def main(config):
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)

    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    try:
        if (
            is_cuda_available
            and config.global_profiler.tool == "nsys"
            and OmegaConf.select(config.global_profiler, "steps") is not None
            and len(OmegaConf.select(config.global_profiler, "steps")) > 0
        ):
            nsight_options = OmegaConf.to_container(
                config.global_profiler.global_tool_config.nsys.controller_nsight_options
            )
            runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
        else:
            runner = TaskRunner.remote()
        ray.get(runner.run.remote(config))
    finally:
        if ray.is_initialized():
            ray.shutdown()


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """
    DAPO TaskRunner 与 PPO TaskRunner 结构类似。

    ====== TaskRunner 的作用 ======

    TaskRunner 是 Ray Actor，运行在独立进程中：
    - 协调 DAPO 训练流程
    - 创建 WorkerGroup
    - 不占用 GPU（只做调度）

    ====== main_dapo.py TaskRunner vs main_ppo.py TaskRunner ======

    | 差异点 | main_ppo.py TaskRunner | main_dapo.py TaskRunner |
    |--------|------------------------|-------------------------|
    | Trainer 类 | RayPPOTrainer | RayDAPOTrainer |
    | Worker 选择 | 同 | 同 |
    | 数据集创建 | 同 | 同 |
    | 训练循环 | PPO 9 步固定循环 | DAPO 动态采样循环 |

    ====== run() 方法流程 ======

    1. 解析配置
    2. 加载 tokenizer
    3. 创建 Worker 类映射（Actor, Critic, RM）
    4. 创建资源池
    5. 创建数据集
    6. 创建 RayDAPOTrainer
    7. RayDAPOTrainer.fit() → DAPO 训练
    """

    def run(self, config):
        """执行 DAPO 训练流程。

        ====== run() 流程 ======

        Step 1: 解析配置
        Step 2: 加载 tokenizer
        Step 3: 创建 Worker 类映射
        Step 4: 创建资源池
        Step 5: 创建数据集
        Step 6: 创建 RayDAPOTrainer
        Step 7: RayDAPOTrainer.fit()

        ====== 与 main_ppo.py run() 的差异 ======

        主要差异在于：
        - 创建 RayDAPOTrainer 而非 RayPPOTrainer
        - RayDAPOTrainer 继承 RayPPOTrainer，重写 fit() 实现动态采样
        """
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        # 打印当前进程信息
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        # pprint(): 打印完整配置
        # resolve=True: 解析所有引用（如 ${actor.path}）
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values

        # resolve(): 解析配置中的所有插值引用
        OmegaConf.resolve(config)

        # ====== Step 2: 加载模型 ======
        # download the checkpoint from hdfs
        # copy_to_local(): 从远程存储下载模型到本地
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # ====== Step 3: 加载 Tokenizer ======
        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        # trust_remote_code: 是否执行模型仓库中的自定义代码
        trust_remote_code = config.data.get("trust_remote_code", False)

        # hf_tokenizer(): 加载 HuggingFace tokenizer
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # used for multimodal LLM, could be none
        # processor: 多模态模型需要 processor（如图像预处理）
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # ====== Step 4: 创建 Worker 类映射 ======
        from verl.single_controller.ray import RayWorkerGroup

        # define worker classes
        # DAPO Worker 选择与 PPO 相同
        # 根据 strategy（FSDP/Megatron）选择 Worker 类
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}

            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(AsyncActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(
            config,
            tokenizer,
            0,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )

        # Note that we always use function-based RM for validation
        val_reward_fn = load_reward_manager(
            config,
            tokenizer,
            1,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
