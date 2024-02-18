# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import functools
import os
import subprocess
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch import distributed as torch_dist
from torch.distributed import ProcessGroup

from collections.abc import Iterable, Mapping

_LOCAL_PROCESS_GROUP = None


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def smart_DDP(model):
    # Model DDP creation with checks
    assert not torch.__version__ == '1.12.0', \
        'torch==1.12.0 torchvision==0.13.0 DDP training is not supported due to a known issue. ' \
        'Please upgrade or downgrade torch to use DDP. See https://github.com/ultralytics/yolov5/issues/8395'
    if torch.__version__ == '1.11.0':
        return torch.nn.parallel.DistributedDataParallel(model,
                                                         device_ids=[LOCAL_RANK],
                                                         output_device=LOCAL_RANK,
                                                         static_graph=True)
    else:
        return torch.nn.parallel.DistributedDataParallel(model,
                                                         device_ids=[LOCAL_RANK],
                                                         output_device=LOCAL_RANK)


def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""
    return torch_dist.is_available() and torch_dist.is_initialized()


def get_local_group() -> Optional[ProcessGroup]:
    """Return local process group."""
    if not is_distributed():
        return None

    if _LOCAL_PROCESS_GROUP is None:
        raise RuntimeError('Local process group is not created, please use '
                           '`init_local_group` to setup local process group.')

    return _LOCAL_PROCESS_GROUP


def get_default_group() -> Optional[ProcessGroup]:
    """Return default process group."""

    return torch_dist.distributed_c10d._get_default_group()


def infer_launcher():
    if 'WORLD_SIZE' in os.environ:
        return 'pytorch'
    elif 'SLURM_NTASKS' in os.environ:
        return 'slurm'
    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        return 'mpi'
    else:
        return 'none'


def init_local_group(node_rank: int, num_gpus_per_node: int):
    """Setup the local process group.

    Setup a process group which only includes processes that on the same
    machine as the current process.

    The code is modified from
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py

    Args:
        node_rank (int): Rank of machines used for training.
        num_gpus_per_node (int): Number of gpus used for training in a single
            machine.
    """  # noqa: W501
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None

    ranks = list(
        range(node_rank * num_gpus_per_node,
              (node_rank + 1) * num_gpus_per_node))
    _LOCAL_PROCESS_GROUP = torch_dist.new_group(ranks)


def get_backend(group: Optional[ProcessGroup] = None) -> Optional[str]:
    """Return the backend of the given process group.

    Note:
        Calling ``get_backend`` in non-distributed environment will return
        None.

    Args:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific
            group is specified, the calling process must be part of
            :attr:`group`. Defaults to None.

    Returns:
        str or None: Return the backend of the given process group as a lower
        case string if in distributed environment, otherwise None.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_backend(group)
    else:
        return None


def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    """Return the number of the given process group.

    Note:
        Calling ``get_world_size`` in non-distributed environment will return
        1.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the number of processes of the given process group if in
        distributed environment, otherwise 1.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_world_size(group)
    else:
        return 1


def get_rank(group: Optional[ProcessGroup] = None) -> int:
    """Return the rank of the given process group.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Note:
        Calling ``get_rank`` in non-distributed environment will return 0.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the rank of the process group if in distributed
        environment, otherwise 0.
    """

    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_rank(group)
    else:
        return 0


def get_local_size() -> int:
    """Return the number of the current node.

    Returns:
        int: Return the number of processes in the current node if in
        distributed environment, otherwise 1.
    """
    if not is_distributed():
        return 1

    if _LOCAL_PROCESS_GROUP is None:
        raise RuntimeError('Local process group is not created, please use '
                           '`init_local_group` to setup local process group.')

    return torch_dist.get_world_size(_LOCAL_PROCESS_GROUP)




def get_dist_info(group: Optional[ProcessGroup] = None) -> Tuple[int, int]:
    """Get distributed information of the given process group.

    Note:
        Calling ``get_dist_info`` in non-distributed environment will return
        (0, 1).

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        tuple[int, int]: Return a tuple containing the ``rank`` and
        ``world_size``.
    """
    world_size = get_world_size(group)
    rank = get_rank(group)
    return rank, world_size


def is_main_process(group: Optional[ProcessGroup] = None) -> bool:
    """Whether the current rank of the given process group is equal to 0.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        bool: Return True if the current rank of the given process group is
        equal to 0, otherwise False.
    """
    return get_rank(group) == 0


def master_only(func: Callable) -> Callable:
    """Decorate those methods which should be executed in master process.

    Args:
        func (callable): Function to be decorated.

    Returns:
        callable: Return decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)

    return wrapper


def barrier(group: Optional[ProcessGroup] = None) -> None:
    """Synchronize all processes from the given process group.

    This collective blocks processes until the whole group enters this
    function.

    Note:
        Calling ``barrier`` in non-distributed environment will do nothing.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        torch_dist.barrier(group)


def get_data_device(data: Union[Tensor, Mapping, Iterable]) -> torch.device:
    """Return the device of ``data``.

    If ``data`` is a sequence of Tensor, all items in ``data`` should have a
    same device type.

    If ``data`` is a dict whose values are Tensor, all values should have a
    same device type.

    Args:
        data (Tensor or Sequence or dict): Inputs to be inferred the device.

    Returns:
        torch.device: The device of ``data``.

    Examples:
        >>> import torch
        >>> from mmengine.dist import cast_data_device
        >>> # data is a Tensor
        >>> data = torch.tensor([0, 1])
        >>> get_data_device(data)
        device(type='cpu')
        >>> # data is a list of Tensor
        >>> data = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        >>> get_data_device(data)
        device(type='cpu')
        >>> # data is a dict
        >>> data = {'key1': torch.tensor([0, 1]), 'key2': torch.tensor([0, 1])}
        >>> get_data_device(data)
        device(type='cpu')
    """
    if isinstance(data, Tensor):
        return data.device
    elif isinstance(data, Mapping):
        pre = None
        for v in data.values():
            cur = get_data_device(v)
            if pre is None:
                pre = cur
            else:
                if cur != pre:
                    raise ValueError(
                        'device type in data should be consistent, but got '
                        f'{cur} and {pre}')
        if pre is None:
            raise ValueError('data should not be empty.')
        return pre
    elif isinstance(data, Iterable) and not isinstance(data, str):
        pre = None
        for item in data:
            cur = get_data_device(item)
            if pre is None:
                pre = cur
            else:
                if cur != pre:
                    raise ValueError(
                        'device type in data should be consistent, but got '
                        f'{cur} and {pre}')
        if pre is None:
            raise ValueError('data should not be empty.')
        return pre
    else:
        raise TypeError('data should be a Tensor, sequence of tensor or dict, '
                        f'but got {data}')


def get_comm_device(group: Optional[ProcessGroup] = None) -> torch.device:
    """Return the device for communication among groups.

    Args:
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        torch.device: The device of backend.
    """
    backend = get_backend(group)
    if backend == 'hccl':
        import torch_npu  # noqa: F401
        return torch.device('npu', torch.npu.current_device())
    elif backend == torch_dist.Backend.NCCL:
        return torch.device('cuda', torch.cuda.current_device())
    elif backend == 'cncl':
        import torch_mlu  # noqa: F401
        return torch.device('mlu', torch.mlu.current_device())
    elif backend == 'smddp':
        return torch.device('cuda', torch.cuda.current_device())
    else:
        # GLOO and MPI backends use cpu device by default
        return torch.device('cpu')


def cast_data_device(
    data: Union[Tensor, Mapping, Iterable],
    device: torch.device,
    out: Optional[Union[Tensor, Mapping, Iterable]] = None
) -> Union[Tensor, Mapping, Iterable]:
    """Recursively convert Tensor in ``data`` to ``device``.

    If ``data`` has already on the ``device``, it will not be casted again.

    Args:
        data (Tensor or list or dict): Inputs to be casted.
        device (torch.device): Destination device type.
        out (Tensor or list or dict, optional): If ``out`` is specified, its
            value will be equal to ``data``. Defaults to None.

    Returns:
        Tensor or list or dict: ``data`` was casted to ``device``.
    """
    if out is not None:
        if type(data) != type(out):
            raise TypeError(
                'out should be the same type with data, but got data is '
                f'{type(data)} and out is {type(data)}')

        if isinstance(out, set):
            raise TypeError('out should not be a set')

    if isinstance(data, Tensor):
        if get_data_device(data) == device:
            data_on_device = data
        else:
            data_on_device = data.to(device)

        if out is not None:
            # modify the value of out inplace
            out.copy_(data_on_device)  # type: ignore

        return data_on_device
    elif isinstance(data, Mapping):
        data_on_device = {}
        if out is not None:
            data_len = len(data)
            out_len = len(out)  # type: ignore
            if data_len != out_len:
                raise ValueError('length of data and out should be same, '
                                 f'but got {data_len} and {out_len}')

            for k, v in data.items():
                data_on_device[k] = cast_data_device(v, device,
                                                     out[k])  # type: ignore
        else:
            for k, v in data.items():
                data_on_device[k] = cast_data_device(v, device)

        if len(data_on_device) == 0:
            raise ValueError('data should not be empty')

        # To ensure the type of output as same as input, we use `type(data)`
        # to wrap the output
        return type(data)(data_on_device)  # type: ignore
    elif isinstance(data, Iterable) and not isinstance(
            data, str) and not isinstance(data, np.ndarray):
        data_on_device = []
        if out is not None:
            for v1, v2 in zip(data, out):
                data_on_device.append(cast_data_device(v1, device, v2))
        else:
            for v in data:
                data_on_device.append(cast_data_device(v, device))

        if len(data_on_device) == 0:
            raise ValueError('data should not be empty')

        return type(data)(data_on_device)  # type: ignore
    else:
        raise TypeError('data should be a Tensor, list of tensor or dict, '
                        f'but got {data}')


def sync_random_seed(group: Optional[ProcessGroup] = None) -> int:
    """Synchronize a random seed to all processes.

    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Random seed.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>> # non-distributed environment
        >>> seed = dist.sync_random_seed()
        >>> seed  # which a random number
        587791752

        >>> distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> seed = dist.sync_random_seed()
        >>> seed
        587791752  # Rank 0
        587791752  # Rank 1
    """
    seed = np.random.randint(2**31)
    if get_world_size(group) == 1:
        return seed

    if group is None:
        group = get_default_group()

    backend_device = get_comm_device(group)

    if get_rank(group) == 0:
        random_num = torch.tensor(seed, dtype=torch.int32).to(backend_device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32).to(backend_device)

    torch_dist.broadcast(random_num, src=0, group=group)

    return random_num.item()
