"""Simplified parallel state helpers."""
from dataclasses import dataclass


@dataclass
class ProcessGroup:
    """Represents a parallel process group."""
    group: str | None = None
    rank: int = 0
    size: int = 1


_tensor_model_parallel_world_size = 1
_data_parallel_world_size = 1
_moe_expert_parallel_world_size = 1
_moe_tensor_parallel_world_size = 1
_pipeline_model_parallel_world_size = 1
_pipeline_model_parallel_rank = 0
_data_parallel_group = ProcessGroup()


def set_parallel_sizes(
        *,
        tp: int | None = None,
        dp: int | None = None,
        moe_tp: int | None = None,
        moe_ep: int | None = None,
        pp: int | None = None,
        pp_rank: int | None = None
):
    global _tensor_model_parallel_world_size
    global _data_parallel_world_size
    global _moe_expert_parallel_world_size
    global _moe_tensor_parallel_world_size
    global _pipeline_model_parallel_world_size
    global _pipeline_model_parallel_rank

    if tp is not None:
        _tensor_model_parallel_world_size = tp
    if dp is not None:
        _data_parallel_world_size = dp
    if moe_tp is not None:
        _moe_tensor_parallel_world_size = moe_tp
    if moe_ep is not None:
        _moe_expert_parallel_world_size = moe_ep
    if pp is not None:
        _pipeline_model_parallel_world_size = pp
    if pp_rank is not None:
        _pipeline_model_parallel_rank = pp_rank


def get_tensor_model_parallel_world_size():
    return _tensor_model_parallel_world_size


def get_data_parallel_world_size():
    return _data_parallel_world_size


def get_moe_expert_parallel_world_size():
    return _moe_expert_parallel_world_size


def get_moe_tensor_parallel_world_size():
    return _moe_tensor_parallel_world_size


def get_pipeline_model_parallel_world_size():
    return _pipeline_model_parallel_world_size


def get_pipeline_model_parallel_rank():
    return _pipeline_model_parallel_rank


def get_data_parallel_group():
    return _data_parallel_group


def initialize_model_parallel(*, tensor_model_parallel_size):
    set_parallel_sizes(tp=tensor_model_parallel_size)

