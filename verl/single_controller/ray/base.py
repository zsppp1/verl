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
import inspect
import logging
import os
import socket
from copy import deepcopy
from typing import Any, Optional

import numpy as np
import ray
from ray.experimental.state.api import get_actor
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

from verl.protocol import DataProto, _padding_size_key
from verl.single_controller.base import ClassWithInitArgs, ResourcePool, Worker, WorkerGroup
from verl.single_controller.base.decorator import MAGIC_ATTR, Dispatch
from verl.utils.py_functional import temp_env_var

__all__ = ["Worker"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def get_random_string(length: int) -> str:
    import random
    import string

    letters_digits = string.ascii_letters + string.digits
    return "".join(random.choice(letters_digits) for _ in range(length))


def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    class Functor:
        def __call__(this, *args, **kwargs):
            args, kwargs = dispatch_fn(self, *args, **kwargs)
            padding_count = kwargs.pop(_padding_size_key, 0)
            output = execute_fn(method_name, *args, **kwargs)
            if blocking:
                output = ray.get(output)
            output = collect_fn(self, output)
            if padding_count > 0:
                if isinstance(output, DataProto):
                    indices = [i for i in range(len(output))][:-padding_count]
                    output = output.select_idxs(indices)
                elif isinstance(output, list):
                    output = output[:-padding_count]
            return output

    # use class type to pass the method_name to get a better observability
    return type(method_name, (Functor,), {})()


def sort_placement_group_by_node_ip(pgs: list[PlacementGroup]) -> list[PlacementGroup]:
    """
    Sort the placement groups by node ip, all bundles in a single placement group should be on the same node.

    FSDPCheckpointManager saves sharded model states and optimizer states in local storage, which requires RANK
    to be consistent across nodes when resume from checkpoint.

    With this function, if there's only one resource pool and there's no node change, RANK should be consistent
    across nodes in multiple ray jobs, even if the whole ray cluster is restarted.
    """
    node_ip = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes()}
    pg_ip = {}
    for pg in pgs:
        specs = ray._private.state.state.placement_group_table(pg.id)
        # all bunles should be on the same node
        node_id = specs["bundles_to_node_id"][0]
        pg_ip[pg.id] = node_ip[node_id]
    return sorted(pgs, key=lambda pg: pg_ip[pg.id])


@ray.remote
def get_master_addr_port() -> tuple[str, str]:
    addr = ray.util.get_node_ip_address().strip("[]")
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    return addr, str(port)


class RayResourcePool(ResourcePool):
    """
    ====== RayResourcePool 是什么？======

    RayResourcePool 是 GPU 资源池管理类，用于：
    1. 抽象物理 GPU 资源
    2. 创建 Ray PlacementGroup（资源调度单位）
    3. 分配 GPU 给 Worker

    ====== 核心概念 ======

    process_on_nodes: 每个节点有多少个进程/GPU
    例如 [8, 8] 表示两个节点，每个节点 8 个 GPU

    max_colocate_count: 一个 GPU 可以复用多少次
    用于 GPU 时间分片，让多个 Worker 共享同一个物理 GPU

    ====== PlacementGroup 是什么？======

    PlacementGroup 是 Ray 的资源调度单位，一组"bundle"的集合。
    每个 bundle = 1 GPU + N CPU

    例如 process_on_nodes=[8, 8] 会创建两个 PlacementGroup：
    - PG_0: 8 个 bundle，分布在 Node 1
    - PG_1: 8 个 bundle，分布在 Node 2

    ====== 数据流示例 ======

    配置阶段：
      resource_pool_spec = {'global_pool': [8, 8]}  # 两个节点各 8 GPU
      → ResourcePoolManager 创建两个 RayResourcePool

    Worker 分配阶段：
      Actor WorkerGroup 需要 16 个 Worker
      → 每个 Worker 占用一个 bundle（1 GPU）
      → Worker_0 用 PG_0 bundle_0，Worker_1 用 PG_0 bundle_1，...

    ====== 使用示例 ======

    # 创建资源池
    pool = RayResourcePool(
        process_on_nodes=[8, 8],  # 16 GPU total
        use_gpu=True,
        max_colocate_count=5      # 每个 GPU 可复用 5 次
    )

    # 获取 PlacementGroup
    pgs = pool.get_placement_groups(strategy="STRICT_PACK")
    # 返回 2 个 PG，各 8 个 bundle

    # 创建 WorkerGroup 时使用
    wg = RayWorkerGroup(resource_pool=pool, ...)
    """

    def __init__(
        self,
        process_on_nodes: Optional[list[int]] = None,
        use_gpu: bool = True,
        name_prefix: str = None,
        max_colocate_count: int = 10,
        detached=False,
        accelerator_type: Optional[str] = None,
    ) -> None:
        """
        Args:
            process_on_nodes: 每个节点的进程数列表
                例如 [8, 8] = 两个节点各 8 进程 = 16 GPU total
            use_gpu: 是否分配 GPU
            name_prefix: PlacementGroup 命名前缀
            max_colocate_count: GPU 复用次数（时间分片）
                默认 10，表示 1 个 GPU 可分给 10 个 Worker
            detached: Actor 是否持久化（不随 session 结束销毁）
            accelerator_type: 特定加速器类型（如 "H100"）
        """
        super().__init__(process_on_nodes, max_colocate_count)
        self.use_gpu = use_gpu
        # name_prefix 用于 PlacementGroup 命名，方便调试
        self.name_prefix = get_random_string(length:6) if name_prefix is None else name_prefix
        self.pgs = None  # PlacementGroup 列表，创建后缓存
        self.detached = detached
        self.accelerator_type = accelerator_type

    def get_placement_groups(self, strategy="STRICT_PACK", name=None, device_name="cuda"):
        """
        创建或返回 PlacementGroup 列表

        ====== strategy 参数 ======

        | Strategy | 含义 | 使用场景 |
        |----------|------|----------|
        | STRICT_PACK | 强制打包，所有 bundle 在同一节点 | FSDP/DeepSpeed 需要节点内通信 |
        | PACK | 尽量打包 | 默认策略 |
        | SPREAD | 尽量分散 | 容错性高 |
        | STRICT_SPREAD | 强制分散 | 高可用场景 |

        ====== bundle 结构 ======

        每个 bundle = {
            "CPU": max_colocate_count,  # CPU 数量
            "GPU": 1,                   # GPU 数量
            "accelerator_type": 1e-4    # 特定 GPU 类型（可选）
        }

        ====== 返回示例 ======

        process_on_nodes=[8, 8] 返回：
        [
            PlacementGroup_0: 8 bundles on Node_1,
            PlacementGroup_1: 8 bundles on Node_2
        ]
        """
        # 如果已经创建过，直接返回缓存的
        if self.pgs is not None:
            return self.pgs

        # 生成 PlacementGroup 名称前缀
        pg_name_prefix = (
            name if name else f"{self.name_prefix}verl_group_{'_'.join([str(count) for count in self._store])}:"
        )

        # 设备类型转换：cuda → GPU, npu → NPU
        if device_name == "npu":
            device_name = "NPU"
        elif device_name == "cuda":
            device_name = "GPU"

        # 定义每个 bundle 的资源
        bundle = {"CPU": self.max_colocate_count}
        if self.use_gpu:
            bundle[device_name] = 1  # 每个 bundle 有 1 个 GPU
            if self.accelerator_type is not None:
                # 特定 GPU 类型（如 H100），用小数值作为标记
                bundle[self.accelerator_type] = 1e-4

        # 根据 process_on_nodes 创建 bundle 列表
        # process_on_nodes=[8, 8] → pg_scheme=[[bundle×8], [bundle×8]]
        pg_scheme = [[bundle.copy() for _ in range(process_count)] for process_count in self._store]

        # lifetime: detached 模式下 PG 持久化
        lifetime = "detached" if self.detached else None

        # 创建 PlacementGroup
        pgs = [
            placement_group(bundles=bundles, strategy=strategy, name=pg_name_prefix + str(idx), lifetime=lifetime)
            for idx, bundles in enumerate(pg_scheme)
        ]

        # 等待所有 PG 就绪（资源分配完成）
        ray.get([pg.ready() for pg in pgs])

        # 按 node IP 排序，确保 checkpoint 恢复时 RANK 一致
        self.pgs = sort_placement_group_by_node_ip(pgs)
        return pgs


class SubRayResourcePool(RayResourcePool):
    def __init__(
        self,
        placement_groups: list[PlacementGroup],
        start_bundle_index: int,
        subgroup_world_size: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pgs = placement_groups
        self.start_bundle_index = start_bundle_index
        self.subgroup_world_size = subgroup_world_size

    @property
    def world_size(self):
        return self.subgroup_world_size


def extract_pg_from_exist(
    resource_pools: dict[str, RayResourcePool], src_role_names: list[str], resource_pool: RayResourcePool
) -> list:
    src_pgs = [
        pg
        for role_name, resource_pool in resource_pools.items()
        for pg in resource_pool.get_placement_groups()
        if role_name in src_role_names
    ]

    sorted_src_pgs = sorted(src_pgs, key=lambda pg: pg.bundle_count, reverse=True)
    sorted_process_on_nodes = sorted([(val, idx) for idx, val in enumerate(resource_pool.store)], reverse=True)

    unsorted_pgs: list[tuple[int, PlacementGroup]] = []
    searching_idx = 0
    for request_process, original_idx in sorted_process_on_nodes:
        assert searching_idx < len(sorted_src_pgs), f"no enough nodes for request: searching {searching_idx} th node"
        assert request_process <= sorted_src_pgs[searching_idx].bundle_count, (
            f"requesting {request_process} processes, bundle count cannot satisfy"
        )
        unsorted_pgs.append((original_idx, sorted_src_pgs[searching_idx]))
        searching_idx += 1

    return [pg for _, pg in sorted(unsorted_pgs)]


# split a RayResourcePool or SubRayResourcePool into multiple SubRayResourcePool
def split_resource_pool(
    resource_pool: RayResourcePool | SubRayResourcePool, split_size: int | list[int]
) -> list[SubRayResourcePool]:
    """
    Split a RayResourcePool into multiple SubRayResourcePool.
    resouce_pool can also be a SubRayResourcePool (have been splited) for multiple-time spliting.

    Args:
        resource_pool (RayResourcePool | SubRayResourcePool): The resource pool to split.
        split_size (int | list[int]): The size of each split. If int, all splits will have the same size.
            If list[int], each element in the list represents the size of a split.

    Returns:
        list[SubRayResourcePool]: A list of SubRayResourcePool after splitting.
    """
    # convert split_size to list[int]
    if isinstance(split_size, int):
        assert resource_pool.world_size % split_size == 0, "split_size must be a divisor of world_size"
        num_replica = resource_pool.world_size // split_size
        split_size_list = [split_size] * num_replica
    else:
        split_size_list = split_size

    assert sum(split_size_list) == resource_pool.world_size, "split_size must sum up to world_size"

    # judge if this resource pool has been splited
    if isinstance(resource_pool, SubRayResourcePool):
        start_bundle_idx_list = np.cumsum([resource_pool.start_bundle_index] + split_size_list[:-1])
    else:
        start_bundle_idx_list = np.cumsum([0] + split_size_list[:-1])

    # ensure resource_pool.pgs has been initialized
    placement_groups = resource_pool.get_placement_groups()
    split_resource_pools = [
        SubRayResourcePool(
            process_on_nodes=resource_pool.store,
            use_gpu=resource_pool.use_gpu,
            name_prefix=f"{resource_pool.name_prefix}_split_{split_idx}",
            max_colocate_count=resource_pool.max_colocate_count,
            placement_groups=placement_groups,
            start_bundle_index=start_bundle_idx_list[split_idx],
            subgroup_world_size=split_size_list[split_idx],
        )
        for split_idx in range(len(split_size_list))
    ]
    return split_resource_pools


def merge_resource_pool(rp1: RayResourcePool, rp2: RayResourcePool) -> RayResourcePool:
    assert rp1.use_gpu == rp2.use_gpu, "Both RayResourcePool must either use_gpu or not"
    assert rp1.max_colocate_count == rp2.max_colocate_count, "Both RayResourcePool must has the same max_colocate_count"
    assert rp1.n_gpus_per_node == rp2.n_gpus_per_node, "Both RayResourcePool must has the same n_gpus_per_node"
    assert rp1.detached == rp2.detached, "Detached ResourcePool cannot be merged with non-detached ResourcePool"

    new_store = rp1.store + rp2.store

    merged = type(rp1)(new_store, rp1.use_gpu, f"{rp1.name_prefix}_{rp2.name_prefix}")
    merged.pgs = rp1.get_placement_groups() + rp2.get_placement_groups()

    return merged


class RayClassWithInitArgs(ClassWithInitArgs):
    """A wrapper class for Ray actors with initialization arguments.

    This class extends ClassWithInitArgs to provide additional functionality for
    configuring and creating Ray actors with specific resource requirements and
    scheduling strategies.
    """

    def __init__(self, cls, *args, **kwargs) -> None:
        # self._options = kwargs.pop('options', dict())
        super().__init__(cls, *args, **kwargs)
        self._options = {}
        self._additional_resource = {}

    def set_additional_resource(self, additional_resource):
        """Set additional resource requirements for the actor.

        Args:
            additional_resource: Dictionary specifying additional resource requirements
        """
        self._additional_resource = additional_resource

    def update_options(self, options: dict):
        """Update the Ray actor creation options.

        Args:
            options: Dictionary of options to update
        """
        self._options.update(options)

    def __call__(
        self,
        placement_group,
        placement_group_bundle_idx,
        use_gpu: bool = True,
        num_gpus=1,
        sharing_with=None,
        device_name="cuda",
    ) -> Any:
        """Create and return a Ray actor with the configured options.

        Args:
            placement_group: Ray placement group for scheduling
            placement_group_bundle_idx: Index of the bundle in the placement group
            use_gpu: Whether to use GPU resources
            num_gpus: Number of GPUs to allocate
            sharing_with: Actor to share resources with
            device_name: Device for training

        Returns:
            A Ray actor handle with the configured options
        """
        if sharing_with is not None:
            target_node_id = ray.get(sharing_with.get_node_id.remote())
            visible_devices = ray.get(sharing_with.get_cuda_visible_devices.remote())
            options = {"scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=target_node_id, soft=False)}
            return self.cls.options(**options).remote(*self.args, cuda_visible_devices=visible_devices, **self.kwargs)

        options = {
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=placement_group, placement_group_bundle_index=placement_group_bundle_idx
            )
        }
        options.update(self._options)

        if use_gpu and device_name == "cuda":
            options["num_gpus"] = num_gpus
        if use_gpu and device_name == "npu":
            options["resources"] = {"NPU": num_gpus}

        if len(self._additional_resource) > 1:
            for k, v in self._additional_resource.items():
                options[k] = v

        # print("cls:", self.cls)
        # print("args: ", self.args)
        # print("kwargs: ", self.kwargs)
        return self.cls.options(**options).remote(*self.args, **self.kwargs)


class RayWorkerGroup(WorkerGroup):
    """A group of Ray workers that can be managed collectively.

    This class extends WorkerGroup to provide Ray-specific functionality for
    creating and managing groups of Ray actors with specific resource requirements
    and scheduling strategies.

    ====== RayWorkerGroup 是什么？======

    RayWorkerGroup 管理一组 Ray Actor（Worker），提供：
    1. 创建多个 Worker 并分配 GPU
    2. RPC 调用 Worker 的方法
    3. 收集多个 Worker 的结果

    ====== Worker 是什么？======

    Worker = Ray.remote 装饰的类 = 独立的 Python 进程

    每个 Worker 运行在独立的进程中，有自己的：
    - GPU 资源（通过 PlacementGroup 分配）
    - 模型副本
    - 内存空间

    ====== RayWorkerGroup 结构示例 ======

    RayWorkerGroup(
        resource_pool=RayResourcePool,  # GPU 资源来源
        ray_cls_with_init=ActorRolloutRefWorker,  # Worker 类
        world_size=16  # 16 个 Worker
    )

    内部结构：
    ┌─────────────────────────────────────────────────────┐
    │  RayWorkerGroup                                     │
    │  ─────────────────                                  │
    │                                                     │
    │  _workers: [Worker_0, Worker_1, ..., Worker_15]    │
    │                                                     │
    │  每个 Worker：                                      │
    │  - 独立进程                                         │
    │  - 占用 1 个 GPU                                    │
    │  - 持有模型副本                                     │
    │  - 通过 RPC 被调用                                  │
    │                                                     │
    │  方法调用：                                         │
    │  generate_sequences(batch) →                       │
    │    Worker_0.generate(batch[0])                     │
    │    Worker_1.generate(batch[1])                     │
    │    ...                                              │
    │    Worker_15.generate(batch[15])                   │
    │  → 收集所有结果 → concat → 返回                    │
    └─────────────────────────────────────────────────────┘

    ====== 关键方法 ======

    | 方法 | 作用 |
    |------|------|
    | generate_sequences() | 调用所有 Worker 生成 |
    | compute_log_prob() | 调用所有 Worker 计算 log prob |
    | compute_values() | 调用所有 Worker 计算 values |
    | update_actor() | 调用所有 Worker 更新 Actor |
    | update_critic() | 调用所有 Worker 更新 Critic |

    ====== RPC 调用流程 ======

    1. dispatch: 将 batch 分发给各个 Worker
    2. execute: RPC 调用 Worker.method.remote()
    3. collect: ray.get() 获取结果，concat 合并

    ====== 使用示例 ======

    # 创建 WorkerGroup
    actor_wg = RayWorkerGroup(
        resource_pool=global_pool,
        ray_cls_with_init=RayClassWithInitArgs(ActorRolloutRefWorker, ...)
    )

    # 调用方法（自动分发到所有 Worker）
    output = actor_wg.generate_sequences(batch)  # batch 分成 16份，并行生成
    """

    def __init__(
        self,
        resource_pool: RayResourcePool = None,
        ray_cls_with_init: RayClassWithInitArgs = None,
        bin_pack: bool = True,
        name_prefix: str = None,
        detached=False,
        worker_names=None,
        worker_handles: list[ray.actor.ActorHandle] = None,
        ray_wait_register_center_timeout: int = 300,
        **kwargs,
    ) -> None:
        """Initialize a RayWorkerGroup.

        Args:
            resource_pool: GPU 资源池，包含 PlacementGroup
            ray_cls_with_init: Worker 类 + 初始化参数
                例如：RayClassWithInitArgs(ActorRolloutRefWorker, model_path, ...)
            bin_pack: 是否使用 STRICT_PACK 策略
            name_prefix: Worker 命名前缀（如 "actor_"）
            detached: Worker 是否持久化
            worker_names: 已有 Worker 名称（用于 attach）
            worker_handles: 已有 Worker ActorHandle（用于 attach）
            **kwargs: device_name, profile_steps 等

        ====== 初始化流程 ======

        Step 1: 创建 PlacementGroup（如果 resource_pool.pgs 为空）
        Step 2: 为每个 bundle 创建一个 Worker
        Step 3: 设置环境变量（WORLD_SIZE, RANK, MASTER_ADDR 等）
        Step 4: 绑定 Worker 方法到 WorkerGroup
        """
        # master_addr 和 master_port 用于分布式通信
        self._master_addr = kwargs.pop("master_addr", None)
        self._master_port = kwargs.pop("master_port", None)
        super().__init__(resource_pool=resource_pool, **kwargs)
        self.ray_cls_with_init = ray_cls_with_init
        self.name_prefix = get_random_string(length=6) if name_prefix is None else name_prefix
        self._ray_wait_register_center_timeout = ray_wait_register_center_timeout
        # fused_worker 用于 Actor+Critic 共存场景
        self.fused_worker_used = ray_cls_with_init.fused_worker_used
        self.sub_cls_name = ""
        self.device_name = kwargs.get("device_name", "cuda")
        self.profile_steps = kwargs.get("profile_steps", None)
        self.worker_nsight_options = kwargs.get("worker_nsight_options", None)
        self.customized_worker_env = kwargs.get("worker_env", {})
        if self.worker_nsight_options is not None and self.worker_nsight_options["capture-range-end"] is None:
            self.worker_nsight_options["capture-range-end"] = f"repeat-shutdown:{6 * len(self.profile_steps)}"

        if worker_names is not None and (not self.fused_worker_used):
            assert self._is_init_with_detached_workers
            self._worker_names = worker_names

        if self._is_init_with_detached_workers:
            self._init_with_detached_workers(worker_names=worker_names, worker_handles=worker_handles)
        elif isinstance(resource_pool, SubRayResourcePool):
            self._init_with_subresource_pool(
                resource_pool=resource_pool,
                ray_cls_with_init=ray_cls_with_init,
                bin_pack=bin_pack,
                detached=detached,
                worker_env=self.customized_worker_env,
            )
        else:
            self._init_with_resource_pool(
                resource_pool=resource_pool,
                ray_cls_with_init=ray_cls_with_init,
                bin_pack=bin_pack,
                detached=detached,
                worker_env=self.customized_worker_env,
            )

        if ray_cls_with_init is not None:
            self._bind_worker_method(self.ray_cls_with_init.cls, func_generator)

        self.wg_dict = None
        self.method_names = []

    def _is_worker_alive(self, worker: ray.actor.ActorHandle):
        """Check if a worker actor is still alive.

        Args:
            worker: Ray actor handle to check

        Returns:
            bool: True if the worker is alive, False otherwise
        """
        worker_state_dict = get_actor(worker._actor_id.hex())
        return worker_state_dict.get("state", "undefined") == "ALIVE" if worker_state_dict is not None else False

    def _init_with_detached_workers(self, worker_names, worker_handles):
        # ray.get_actor holds a weak reference to the actor, which causes actors garbage collected unexpectedly
        # if we only hold spawn RayWorkerGroup. By passing actor handle explicitly, spawn RayWorkerGroup have
        # strong reference to these actors.
        # https://github.com/ray-project/ray/pull/45699
        workers = worker_handles if worker_handles else [ray.get_actor(name=name) for name in worker_names]
        self._workers = workers
        self._world_size = len(worker_names)

    def _get_master_addr_port(self, pg):
        """Get master addr and port for this worker group"""
        if self._master_addr is None and self._master_port is None:
            self._master_addr, self._master_port = ray.get(
                get_master_addr_port.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=0
                    ),
                ).remote()
            )
        elif self._master_addr is not None and self._master_port is not None:
            logger.debug(f"{self._master_addr=} {self._master_port=}")
        else:
            raise ValueError(
                "Both 'master_addr' and 'master_port' must be provided if you intend to manually specify them, "
                "or neither should be provided to use Ray's default assignment."
            )

    def _init_with_resource_pool(self, resource_pool, ray_cls_with_init, bin_pack, detached, worker_env=None):
        """Initialize the worker group by creating new workers from a resource pool.

        Args:
            resource_pool: Resource pool for worker allocation
            ray_cls_with_init: Class with initialization arguments for workers
            bin_pack: Whether to use strict bin packing for resource allocation
            detached: Whether workers should be detached
        """
        self.resource_pool = resource_pool

        strategy = "PACK"
        if bin_pack:
            strategy = "STRICT_PACK"
        pgs = resource_pool.get_placement_groups(strategy=strategy, device_name=self.device_name)
        world_size = resource_pool.world_size
        self._world_size = world_size
        # cia.add_kwarg("_world_size", world_size)

        rank = -1
        local_world_size = resource_pool.store[0]
        for pg_idx, pg in enumerate(sort_placement_group_by_node_ip(pgs)):
            assert local_world_size <= pg.bundle_count, f"when generating for {self.name_prefix}, for the "
            if pg_idx == 0:
                self._get_master_addr_port(pg)

            for local_rank in range(local_world_size):
                rank += 1
                self._create_worker(
                    rank=rank,
                    pg_idx=pg_idx,
                    pg=pg,
                    local_rank=local_rank,
                    resource_pool=resource_pool,
                    ray_cls_with_init=ray_cls_with_init,
                    worker_env=worker_env,
                    detached=detached,
                )

    def _init_with_subresource_pool(self, resource_pool, ray_cls_with_init, bin_pack, detached, worker_env=None):
        """Initialize the worker group by creating new workers from a resource pool or sub resource pool.
        Args:
            resource_pool: Resource pool for worker allocation
            ray_cls_with_init: Class with initialization arguments for workers
            bin_pack: Whether to use strict bin packing for resource allocation
            detached: Whether workers should be detached
        """
        strategy = "PACK"
        if bin_pack:
            strategy = "STRICT_PACK"
        pgs = resource_pool.get_placement_groups(strategy=strategy, device_name=self.device_name)
        world_size = resource_pool.world_size
        self._world_size = world_size

        rank = -1
        local_world_size = resource_pool.store[0]
        self._get_master_addr_port(pgs[0])
        for curr_rank in range(resource_pool.start_bundle_index, resource_pool.start_bundle_index + world_size):
            pg_idx = curr_rank // local_world_size
            pg = pgs[pg_idx]
            local_rank = curr_rank % local_world_size
            assert local_world_size <= pg.bundle_count, f"when generating for {self.name_prefix}, for the "

            rank += 1
            self._create_worker(
                rank=rank,
                pg_idx=pg_idx,
                pg=pg,
                local_rank=local_rank,
                resource_pool=resource_pool,
                ray_cls_with_init=ray_cls_with_init,
                worker_env=worker_env,
                detached=detached,
            )

    def _create_worker(self, rank, pg_idx, pg, local_rank, resource_pool, ray_cls_with_init, worker_env, detached):
        world_size = resource_pool.world_size
        use_gpu = resource_pool.use_gpu
        local_world_size = resource_pool.store[0]
        num_gpus = 1 / resource_pool.max_colocate_count

        # we pass in environment variable at option so that Worker can use environment variable to set
        env_vars = {
            "WORLD_SIZE": str(world_size),
            "RANK": str(rank),
            "WG_PREFIX": self.name_prefix,
            "WG_BACKEND": "ray",
            "RAY_LOCAL_WORLD_SIZE": str(local_world_size),
            "MASTER_ADDR": self._master_addr,
            "MASTER_PORT": self._master_port,
        }
        if worker_env is not None:
            logging.debug(f"Appending ray class env, origin: {env_vars}, customized env: {worker_env}")
            conflict_env_vars = set(env_vars.keys()) & set(worker_env.keys())
            if len(conflict_env_vars) > 0:
                logging.error(
                    f"User customized env vars conflict with system env: {conflict_env_vars} "
                    f"Overriding may cause unexpected behavior."
                )
                raise ValueError(f"Cannot override protected system env: {conflict_env_vars}")
            env_vars.update(worker_env)
        import re

        cia_name = type(ray_cls_with_init.cls).__name__
        match = re.search(r"ActorClass\(([^)]+)\)", cia_name)  # ray.remote(Obj) -> "ActorClass(Obj)"
        cia_name = match.group(1) if match else cia_name  # "ActorClass(Obj)" -> "Obj"
        name = f"{self.name_prefix}{cia_name}_{pg_idx}:{local_rank}"  # e.g. Worker_2:5

        if self.profile_steps and self.device_name == "cuda":
            ray_cls_with_init.update_options(
                {
                    "runtime_env": {
                        "env_vars": env_vars,
                        "nsight": self.worker_nsight_options,
                    },
                    "name": name,
                }
            )
        else:
            ray_cls_with_init.update_options({"runtime_env": {"env_vars": env_vars}, "name": name})

        if detached:
            ray_cls_with_init.update_options({"lifetime": "detached"})

        # create a worker
        worker = ray_cls_with_init(
            placement_group=pg,
            placement_group_bundle_idx=local_rank,
            use_gpu=use_gpu,
            num_gpus=num_gpus,
            device_name=self.device_name,
        )
        self._workers.append(worker)
        self._worker_names.append(name)

    @property
    def worker_names(self):
        return self._worker_names

    @classmethod
    def from_detached(
        cls,
        name_prefix=None,
        worker_names=None,
        worker_handles=None,
        ray_cls_with_init=None,
        **kwargs,
    ):
        """Create a worker group from existing detached workers.

        Args:
            name_prefix: Prefix for worker names
            worker_names: Names of existing workers to attach to
            ray_cls_with_init: Class with initialization arguments for workers

        Returns:
            A new RayWorkerGroup instance
        """
        worker_group = cls(
            resource_pool=None,
            ray_cls_with_init=ray_cls_with_init,
            name_prefix=name_prefix,
            worker_names=worker_names,
            worker_handles=worker_handles,
            **kwargs,
        )
        return worker_group

    def spawn(self, prefix_set):
        """Spawn to a dictionary of worker groups, each with a subset of method with prefix.

        Args:
            prefix_set: Set of prefixes to create worker groups for

        Returns:
            Dictionary of worker groups keyed by prefix
        """
        if self.fused_worker_used:
            return self.spawn_fused(prefix_set)

        def _rebind_actor_methods(worker_group, actor_name):
            prefix: str = actor_name + "_"
            for method_name in dir(worker_group):
                if method_name.startswith(prefix):
                    original_method_name = method_name.removeprefix(prefix)
                    method = getattr(worker_group, method_name)
                    setattr(worker_group, original_method_name, method)

        new_worker_group_dict = {}
        for prefix in prefix_set:
            new_worker_group = self.from_detached(
                name_prefix=self.name_prefix,
                worker_names=self._worker_names,
                worker_handles=self._workers,
                ray_cls_with_init=self.ray_cls_with_init,
                profile_steps=self.profile_steps,
                worker_nsight_options=self.worker_nsight_options,
            )

            _rebind_actor_methods(new_worker_group, prefix)
            new_worker_group_dict[prefix] = new_worker_group
        return new_worker_group_dict

    def spawn_fused(self, prefix_set):
        """Create a dictionary of worker groups for fused workers.

        Args:
            prefix_set: Set of prefixes to create worker groups for

        Returns:
            Dictionary of worker groups keyed by prefix
        """
        wg_dict = dict()
        for key in prefix_set:
            new_wg = deepcopy(self)
            new_wg._bind_worker_method(self.ray_cls_with_init.cls.raw_cls_dict[key], func_generator)
            new_wg.sub_cls_name = key
            wg_dict[key] = new_wg
        return wg_dict

    def fuse(self, prefix_set):
        """Fuse multiple worker groups into the current worker group.

        Args:
            prefix_set: Set of prefixes to fuse into the worker group
        """
        if self.wg_dict is None:
            self.wg_dict = self.spawn(prefix_set)
        for role_name, role_wg in self.wg_dict.items():
            setattr(self, role_name, role_wg)
        self.method_names = self._bind_worker_method(self.ray_cls_with_init.cls, func_generator)

    def _execute_remote_single_worker(self, worker, method_name: str, *args, **kwargs):
        """Execute a method on a single worker remotely.

        Args:
            worker: The worker actor handle
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """
        if self.fused_worker_used and method_name not in self.method_names:
            remote_call = getattr(worker, self.fused_worker_execute_fn_name)
            return remote_call.remote(f"{self.sub_cls_name}_fwmn_{method_name}", *args, **kwargs)
        # fused worker not used
        remote_call = getattr(worker, method_name)
        return remote_call.remote(*args, **kwargs)

    def execute_rank_zero_sync(self, method_name: str, *args, **kwargs):
        """Execute a method on rank zero worker synchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Result of the method execution
        """
        return ray.get(self.execute_rank_zero_async(method_name, *args, **kwargs))

    def execute_rank_zero_async(self, method_name: str, *args, **kwargs):
        """Execute a method on rank zero worker asynchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """
        return self._execute_remote_single_worker(self._workers[0], method_name, *args, **kwargs)

    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        """Alias for execute_rank_zero_async.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """
        return self.execute_rank_zero_async(method_name, *args, **kwargs)

    def execute_all(self, method_name: str, *args, **kwargs):
        """Alias for execute_all_async.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of remote object references to the method executions
        """
        return self.execute_all_async(method_name, *args, **kwargs)

    def execute_all_sync(self, method_name: str, *args, **kwargs):
        """Execute a method on all workers synchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of results from all workers
        """
        return ray.get(self.execute_all_async(method_name, *args, **kwargs))

    def execute_all_async(self, method_name: str, *args, **kwargs):
        """Execute a method on all workers asynchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of remote object references to the method executions
        """
        # Here, we assume that if all arguments in args and kwargs are lists,
        # and their lengths match len(self._workers), we'll distribute each
        # element in these lists to the corresponding worker
        # print(f"execute_all_async: method {method_name}({args}, {kwargs})")
        length = len(self._workers)
        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
            if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
                # print(f"splitting args and kwargs into {length} shards")
                result = []
                for i in range(length):
                    sliced_args = tuple(arg[i] for arg in args)
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    result.append(
                        self._execute_remote_single_worker(self._workers[i], method_name, *sliced_args, **sliced_kwargs)
                    )
                return result

        return [self._execute_remote_single_worker(worker, method_name, *args, **kwargs) for worker in self._workers]

    @property
    def master_address(self):
        return self._master_addr

    @property
    def master_port(self):
        return self._master_port

    @property
    def workers(self):
        return self._workers

    @property
    def world_size(self):
        return self._world_size


"""
Utilities that enables creating workers inside the same ray.Actor,
with code written in separate ray.Actors.
"""


# deprecated, switching to FusedWorker
def _bind_workers_method_to_parent(cls, key, user_defined_cls):
    """
    Binds the methods of each worker to the WorkerDict.
    Note that we only bind public methods that are decorated by register
    """

    for method_name in dir(user_defined_cls):
        try:
            method = getattr(user_defined_cls, method_name)
            assert callable(method), f"{method_name} in {user_defined_cls} is not callable"
        except Exception:
            # if it is a property, it will fail because Class doesn't have instance property
            continue

        if hasattr(method, MAGIC_ATTR):

            def generate_function(name, key=key):
                def func(self, *args, **kwargs):
                    # dispatch to the actual worker
                    return getattr(self.worker_dict[key], name)(*args, **kwargs)

                async def async_func(self, *args, **kwargs):
                    # dispatch to the actual worker
                    return await getattr(self.worker_dict[key], name)(*args, **kwargs)

                wrapper = async_func if inspect.iscoroutinefunction(method) else func  # noqa: B023

                return wrapper

            func = generate_function(method_name)
            # pass MAGIC_ATTR for outer worker group
            attrs = getattr(method, MAGIC_ATTR)
            setattr(func, MAGIC_ATTR, attrs)
            try:
                # bind direct rollout method to class without prefix
                if attrs["dispatch_mode"] == Dispatch.DIRECT_ROLLOUT_METHOD and "rollout" in key:
                    assert not hasattr(cls, method_name), (
                        f"conflict direct rollout method {method_name} with role {key}"
                    )
                    setattr(cls, method_name, func)
                    print(f"bind role {key} method {method_name} to class {cls}")
                else:
                    method_name_with_prefix = key + "_" + method_name
                    setattr(cls, method_name_with_prefix, func)
            except Exception as e:
                raise ValueError(f"Fail to set method_name {method_name}") from e


def _unwrap_ray_remote(cls):
    if hasattr(cls, "__ray_actor_class__"):
        cls = cls.__ray_actor_class__
    return cls


def _determine_fsdp_megatron_base_class(mros: list):
    """
    - megatron: base class should be MegatronWorker
    - fsdp: base class should be Worker
    """
    for cls in mros[0]:
        if cls.__name__ == "MegatronWorker":
            return cls
        if cls.__name__ == "Worker":
            return cls
    raise ValueError(f"Cannot determine base class for {mros}")


# deprecated, switching to FusedWorker
def create_colocated_worker_cls(class_dict: dict[str, RayClassWithInitArgs]):
    """
    This function should return a class instance that delegates the calls to every
    cls in cls_dict
    """
    cls_dict = {}
    init_args_dict = {}
    worker_cls = _determine_fsdp_megatron_base_class(
        [cls.cls.__ray_actor_class__.__mro__ for cls in class_dict.values()]
    )
    assert issubclass(worker_cls, Worker), f"worker_cls {worker_cls} should be a subclass of Worker"
    print(f"colocated worker base class {worker_cls}")

    for key, cls in class_dict.items():
        cls_dict[key] = cls.cls
        init_args_dict[key] = {"args": cls.args, "kwargs": cls.kwargs}

    assert cls_dict.keys() == init_args_dict.keys()

    # TODO: create a class with customizable name
    class WorkerDict(worker_cls):
        def __init__(self):
            super().__init__()
            self.worker_dict = {}
            for key, user_defined_cls in cls_dict.items():
                user_defined_cls = _unwrap_ray_remote(user_defined_cls)
                # directly instantiate the class without remote
                # in worker class, e.g. <verl.single_controller.base.worker.Worker>
                # when DISABLE_WORKER_INIT == 1 it will return immediately
                with temp_env_var("DISABLE_WORKER_INIT", "1"):
                    self.worker_dict[key] = user_defined_cls(
                        *init_args_dict[key].get("args", ()), **init_args_dict[key].get("kwargs", {})
                    )

    # now monkey-patch the methods from inner class to WorkerDict
    for key, user_defined_cls in cls_dict.items():
        user_defined_cls = _unwrap_ray_remote(user_defined_cls)
        _bind_workers_method_to_parent(WorkerDict, key, user_defined_cls)

    remote_cls = ray.remote(WorkerDict)
    remote_cls = RayClassWithInitArgs(cls=remote_cls)
    return remote_cls


FusedWorkerCLSName = "FusedWorker"


def create_colocated_worker_raw_cls(class_dict: dict[str, RayClassWithInitArgs]):
    """
    This function returns a FusedWorker class.

    `FusedWorker.{class_name}` -> FusedClass
        Use `class_name` as a param to directly access the underlying class.

    `FusedWorker._fuw_execute("{class_name}_fwmn_{method_name}", *args, **kwargs)`
        First param must be "{class_name}_fwmn_{method_name}" in order to access `method_name`
        of underlying class `{class_name}`.

    `FusedWorker.fused_worker_dict` -> {"class_name": FusedClass}
        Stores all underlying classes.

    `FusedClass.fused_worker_dict` -> {"class_name": FusedClass}
        The same as `FusedWorker.fused_worker_dict`, enables underlying class to access other
        underlying classes.
    """
    raw_cls_dict = {cls_name: _unwrap_ray_remote(cia.cls) for cls_name, cia in class_dict.items()}
    init_args_dict = {cls_name: cia.args for cls_name, cia in class_dict.items()}
    init_kwargs_dict = {cls_name: cia.kwargs for cls_name, cia in class_dict.items()}
    cls_names = list(class_dict.keys())

    # FusedWorker_Actor_Critic
    class_name_renamed = "_".join([FusedWorkerCLSName] + cls_names)

    class FusedWorker(Worker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cls_names = cls_names
            self.raw_cls_dict = raw_cls_dict
            self.init_args_dict = init_args_dict
            self.init_kwargs_dict = init_kwargs_dict

            for cls_name, udc, ud_args, ud_kwargs in zip(
                self.cls_names,
                self.raw_cls_dict.values(),
                self.init_args_dict.values(),
                self.init_kwargs_dict.values(),
                strict=True,
            ):
                with temp_env_var("DISABLE_WORKER_INIT", "1"):
                    udc._get_ray_actor_cls_name = lambda x, name_renamed=class_name_renamed: name_renamed
                    udc._get_ray_method_prefix = lambda x, name_prefixed=cls_name: f"{name_prefixed}_"
                    # cls_name = "actor", "critic", udc = ActorWorker, CriticWorker
                    self.fused_worker_dict[cls_name] = udc(*ud_args, **ud_kwargs)
                    setattr(self, cls_name, self.fused_worker_dict[cls_name])

            # injecting fused_worker to each sub worker so they can be aware of existence of each other
            for _, worker in self.fused_worker_dict.items():
                setattr(worker, Worker.fused_worker_attr_name, self.fused_worker_dict)

        def _fuw_execute(self, method_name: str, *args, **kwargs):
            # for fused_worker, method_name is in a form of "{cls_name}_fwmn_{method_name}"
            # where fwmn stands "fused worker method name"
            names = method_name.split("_fwmn_")
            cls_name = names[0]
            method_name = names[1]

            assert cls_name in self.fused_worker_dict, (
                f"calling {cls_name}'s {method_name}, but {cls_name} not in fused_worker_dict"
            )
            udc_method = getattr(self.fused_worker_dict[cls_name], method_name)
            return udc_method(*args, **kwargs)

    renamed_fused_worker_cls = type(class_name_renamed, (FusedWorker,), {})
    renamed_fused_worker_cls.is_fused_worker = True
    renamed_fused_worker_cls.raw_cls_dict = raw_cls_dict

    return renamed_fused_worker_cls


def create_colocated_worker_cls_fused(class_dict: dict[str, RayClassWithInitArgs]):
    """
    This function returns a RayClassWithInitArgs instance of FusedWorker, which is an replacement
    of `create_colocated_worker_cls`. WorkerGroup constructed using this class will be a colocated
    WorkerGroup, which will be referenced as `ColocateWorkerGroup` below.

    `ColocateWorkerGroup.spawn(prefix_set)`
        returns a dict of WorkerGroup {"class_name": WorkerGroup}, WorkerGroup in this dict will
        have methods of underlying class `class_name` attached.

    `ColocateWorkerGroup.fuse(prefix_set)`
        After executing this function, `ColocateWorkerGroup.{class_name}` will return WorkerGroup
        with methods of underlying class `class_name` attached.
    """
    raw_colocated_worker_cls = create_colocated_worker_raw_cls(class_dict)

    remote_cls = ray.remote(raw_colocated_worker_cls)
    cia = RayClassWithInitArgs(cls=remote_cls)
    cia.fused_worker_used = True

    return cia
