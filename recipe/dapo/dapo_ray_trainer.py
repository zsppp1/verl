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
====== dapo_ray_trainer.py 整体架构 ======

这是 DAPO 算法的 Trainer 实现，继承自 RayPPOTrainer。

====== DAPO vs PPO 对比 ======

| 特点 | PPO | DAPO |
|------|-----|------|
| 裁剪方式 | 对称裁剪 (clip_ratio) | 不对称裁剪 (clip_ratio_low/clip_ratio_high) |
| 采样方式 | 固定 batch | 动态采样（过滤全对/全错） |
| Loss 聚合 | sequence-mean 或 token-mean | token-mean（更稳定） |
| 超长惩罚 | 无 | overlong_buffer 惩罚过长回答 |

====== DAPO 四大创新 ======

1. 不对称裁剪：正负样本不同裁剪力度
   - 正样本：clip_ratio_high（如 2.0）
   - 负样本：clip_ratio_low（如 0.2）
   - 防止负样本过度下降

2. 动态采样：过滤无信息量的样本
   - 同一 prompt 的所有 response 全对 → std=0 → 过滤
   - 同一 prompt 的所有 response 全错 → std=0 → 过滤
   - 保留有差异的样本重新采样

3. Token-level Loss：按 token 而非 sequence 聚合
   - loss_agg_mode = "token-mean"
   - 避免长序列被过度惩罚

4. 超长惩罚：overlong_buffer
   - 对超过长度限制的回答施加惩罚
   - 防止模型生成过长无意义回答

====== RayDAPOTrainer 继承关系 ======

RayDAPOTrainer 继承 RayPPOTrainer：
- 继承：init_workers(), generate_sequences(), update_actor(), update_critic()
- 重写：fit()（实现动态采样循环）

====== 动态采样流程 ======

outer loop (dataloader):
    batch ← dataloader.next()
    ↓
    gen_batch ← repeat(n)
    ↓
    generate_sequences() → responses
    ↓
    compute_reward() → scores
    ↓
    按 uid 分组计算 std
    ↓
    if std=0: continue（过滤）
    ↓
    accumulate batch until batch_size 达到阈值
    ↓
    update_actor(), update_critic()

FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


class RayDAPOTrainer(RayPPOTrainer):
    """
    DAPO Trainer 继承自 RayPPOTrainer。

    ====== 继承关系 ======

    RayDAPOTrainer 继承 RayPPOTrainer：
    - 继承的方法：init_workers(), _update_actor(), _update_critic(), _validate()
    - 重写的方法：fit()（实现 DAPO 动态采样）
    - 新增的方法：compute_kl_related_metrics()

    ====== 为什么重写 fit()？======

    DAPO 需要动态采样：
    - PPO: 固定 batch，每个 batch 都训练
    - DAPO: 动态过滤，std=0 的样本不训练，重新采样

    动态采样需要改变训练循环结构：
    - outer loop: 从 dataloader 获取数据
    - inner loop: 累积足够样本后才更新

    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        """计算 KL 相关的 metrics。

        ====== compute_kl_related_metrics 的作用 ======

        计算 old_log_prob 和 ref_log_prob：
        - old_log_prob: 当前策略的 log probability（用于 PPO ratio）
        - ref_log_prob: 参考策略的 log probability（用于 KL 计算）

        ====== 与 PPO 的差异 ======

        PPO 在 fit() 中计算 old_log_prob，DAPO 单独提取为方法：
        - 方便动态采样时多次调用
        - 计算时机可能不同
        """
        # response_mask: response 部分的 mask
        batch.batch["response_mask"] = compute_response_mask(batch)

        # recompute old_log_probs
        # compute_log_prob(): 计算 old_log_prob
        with marked_timer("old_log_prob", timing_raw, "blue"):
            # actor_rollout_wg.compute_log_prob(): WorkerGroup 计算 log probability
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)

            # entropys: 熵（用于记录 metrics）
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode

            # agg_loss(): 聚合 entropy（token-mean 或 sequence-mean）
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)

            # 记录 entropy metrics
            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)

            # 移除 entropys（不需要存入 batch）
            old_log_prob.batch.pop("entropys")

            # union(): 合入 batch
            batch = batch.union(old_log_prob)

        # compute reference log_prob（如果启用 KL）
        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, "olive"):
                # ref_in_actor: Ref Policy 是否融合在 Actor 中
                if not self.ref_in_actor:
                    # 独立的 Ref Policy Worker
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    # Ref Policy 融合在 Actor Worker 中
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)

                batch = batch.union(ref_log_prob)

        return batch

    def fit(self):
        """
        The training loop of DAPO (Dynamic Advantage Policy Optimization).

        ====== DAPO fit() 的作用 ======

        执行 DAPO 动态采样训练循环：
        1. 从 dataloader 获取数据
        2. 生成 response
        3. 计算 reward
        4. 动态过滤（std=0 的样本）
        5. 累积足够样本后更新模型

        ====== DAPO vs PPO fit() ======

        | PPO fit() | DAPO fit() |
        |-----------|------------|
        | 固定 batch 循环 | 动态采样循环 |
        | 每个 batch 都训练 | 累积 batch 后训练 |
        | 无过滤逻辑 | 过滤 std=0 样本 |
        | 单层循环 | 两层循环（outer: dataloader, inner: accumulate） |

        ====== 动态采样流程 ======

        outer loop (遍历 dataloader):
            batch ← dataloader.next()
            ↓
            gen_batch ← repeat(n)  # 每个 prompt 复制 n 次
            ↓
            generate_sequences() → responses
            ↓
            compute_reward() → scores
            ↓
            按 uid 分组计算 reward std
            ↓
            if std=0: continue（过滤，不累加）
            ↓
            if std>0: 累加到 batch_pool
            ↓
            if batch_pool 达到阈值: 执行训练更新

        ====== 动态过滤原因 ======

        GRPO advantage 计算：A = (r - mean) / std
        - 如果同一 prompt 的所有 response 全对 → reward 全=1 → std=0 → 无法计算
        - 如果同一 prompt 的所有 response 全错 → reward 全=0 → std=0 → 无法计算
        - 这些样本对训练无信息量，应该过滤

        ====== 关键变量 ======

        - batch: 当前累积的样本池（可能跨多个 dataloader batch）
        - num_prompt_in_batch: 累积的 prompt 数量
        - num_gen_batches: 生成次数（用于动态采样限制）

        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        # ====== Step 0: 初始化 logger ======
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # global_steps: 训练步数（每次更新算一步）
        self.global_steps = 0

        # gen_steps: 生成步数（每次生成算一步，可能多次生成才更新一次）
        self.gen_steps = 0

        # ====== Step 1: 加载 checkpoint ======
        # load checkpoint before doing anything
        self._load_checkpoint()

        # ====== Step 2: 执行初始验证 ======
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # ====== Step 3: 配置 rollout skip ======
        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # ====== Step 4: 初始化进度条 ======
        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        # ====== Step 5: 配置 profiling ======
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # ====== Step 6: 初始化动态采样变量 ======
        timing_raw = defaultdict(float)

        # batch: 累积的样本池（可能跨多个 dataloader batch）
        batch = None

        # num_prompt_in_batch: 累积的 prompt 数量（用于判断是否达到阈值）
        num_prompt_in_batch = 0

        # num_gen_batches: 生成次数（用于动态采样限制，防止无限生成）
        num_gen_batches = 0

        # ====== Step 7: 开始动态采样循环 ======
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                # async_rollout: 完成之前的异步 rollout 调用
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                # ====== Step 8: 准备新 batch ======
                # new_batch: 从 dataloader 获取的原始 batch
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1

                # _get_gen_batch(): 分离出生成需要的部分
                gen_batch = self._get_gen_batch(new_batch)
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            # compute reward model score on new_batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(new_batch)
                                new_batch = new_batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(new_batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    if self.config.algorithm.use_kl_in_reward:
                        # We need these metrics for apply_kl_penalty if using kl in reward
                        new_batch = self.compute_kl_related_metrics(new_batch, metrics, timing_raw)
                        # otherwise, we will compute those after dynamic sampling

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    # ====== Step 16: 动态过滤（DAPO 核心）======
                    if not self.config.algorithm.filter_groups.enable:
                        # 如果不启用动态过滤，直接使用 new_batch
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch

                        # ====== Step 16.1: 确定过滤指标 ======
                        # metric_name: 用于过滤的指标名称
                        # - "seq_final_reward": 最终 reward（可能包含 KL penalty）
                        # - "seq_reward": 原始 reward score
                        metric_name = self.config.algorithm.filter_groups.metric

                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            # seq_final_reward: response 的总 reward（token_level_rewards 求和）
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            # seq_reward: response 的原始 score（token_level_scores 求和）
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # ====== Step 16.2: 按 uid 分组计算 std ======
                        # Collect the sequence reward for each trajectory
                        # prompt_uid2metric_vals: uid → metric 列表
                        # 例如：{'p1': [0.9, 0.8, 0.9, 0.8], 'p2': [1.0, 1.0, 1.0, 1.0]}
                        prompt_uid2metric_vals = defaultdict(list)

                        # 遍历所有 response，按 uid 分组
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        # ====== Step 16.3: 计算每个 uid 的 std ======
                        # prompt_uid2metric_std: uid → std
                        # 例如：{'p1': 0.05, 'p2': 0.0}
                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            # np.std(): 计算标准差
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        # ====== Step 16.4: 过滤 std=0 的 uid ======
                        # kept_prompt_uids: 保留的 uid 列表
                        # 条件：std > 0（有差异）或只有 1 个 response（无法计算 std）
                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]

                        # num_prompt_in_batch: 累积保留的 prompt 数量
                        num_prompt_in_batch += len(kept_prompt_uids)

                        # ====== Step 16.5: 过滤 trajectory ======
                        # kept_traj_idxs: 保留的 trajectory 索引
                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        # new_batch[kept_traj_idxs]: 只保留有效的 trajectory
                        new_batch = new_batch[kept_traj_idxs]

                        # ====== Step 16.6: 累积 batch ======
                        # batch: 累积的样本池
                        # DataProto.concat(): 合并多个 DataProto
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        # ====== Step 16.7: 检查是否达到阈值 ======
                        prompt_bsz = self.config.data.train_batch_size

                        # 如果累积的 prompt 数量不足，继续生成
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")

                            # max_num_gen_batches: 最大生成次数限制（防止无限生成）
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches

                            # 如果未达到限制，继续生成下一个 batch
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                self.gen_steps += 1
                                is_last_step = self.global_steps >= self.total_training_steps

                                # continue: 跳到下一个 dataloader batch
                                # 不执行训练更新
                                continue
                            else:
                                # 如果生成次数超过限制，报错
                                # 可能是数据太难，所有样本都被过滤
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === Updating ===
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if not self.config.algorithm.use_kl_in_reward:
                        batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # Compute rollout correction weights and off-policy metrics (inherited from RayPPOTrainer)
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                        # IS and off-policy metrics already have rollout_corr/ prefix
                        metrics.update(is_metrics)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
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
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

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

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
