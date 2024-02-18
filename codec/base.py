# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import numpy as np


def is_method_overridden(method: str, base_class: type,
                         derived_class: Union[type, Any]) -> bool:
    """Check if a method of base class is overridden in derived class.

    Args:
        method (str): the method name to check.
        base_class (type): the class of the base class.
        derived_class (type | Any): the class or instance of the derived class.
    """
    assert isinstance(base_class, type), \
        "base_class doesn't accept instance, Please pass class instead."

    if not isinstance(derived_class, type):
        derived_class = derived_class.__class__

    base_method = getattr(base_class, method)
    derived_method = getattr(derived_class, method)
    return derived_method != base_method


class BaseKeypointCodec(metaclass=ABCMeta):
    """The base class of the keypoint codec.

    A keypoint codec is a module to encode keypoint coordinates to specific
    representation (e.g. heatmap) and vice versa. A subclass should implement
    the methods :meth:`encode` and :meth:`decode`.
    """

    # pass additional encoding arguments to the `encode` method, beyond the
    # mandatory `keypoints` and `keypoints_visible` arguments.
    auxiliary_encode_keys = set()

    field_mapping_table = dict()
    instance_mapping_table = dict()
    label_mapping_table = dict()

    @abstractmethod
    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        """Encode keypoints.

        Note:

            - instance number: N
            - keypoint number: K
            - keypoint dimension: D

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibility in shape
                (N, K, D)

        Returns:
            dict: Encoded items.
        """

    @abstractmethod
    def decode(self, encoded: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoints.

        Args:
            encoded (any): Encoded keypoint representation using the codec

        Returns:
            tuple:
            - keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            - keypoints_visible (np.ndarray): Keypoint visibility in shape
                (N, K, D)
        """

    def batch_decode(self, batch_encoded: Any) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Decode keypoints.

        Args:
            batch_encoded (any): A batch of encoded keypoint
                representations

        Returns:
            tuple:
            - batch_keypoints (List[np.ndarray]): Each element is keypoint
                coordinates in shape (N, K, D)
            - batch_keypoints (List[np.ndarray]): Each element is keypoint
                visibility in shape (N, K)
        """
        raise NotImplementedError()

    @property
    def support_batch_decoding(self) -> bool:
        """Return whether the codec support decoding from batch data."""
        return is_method_overridden('batch_decode', BaseKeypointCodec,
                                    self.__class__)
