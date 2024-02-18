# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Dict, Union, Tuple, List

import numpy as np
import cv2

class LoadImage:
    """Load an image from file or from the np.ndarray in ``results['img']``.

    Required Keys:

        - img_path
        - img (optional)

    Modified Keys:

        - img
        - img_shape
        - ori_shape
        - img_path (optional)

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        backend_args (dict, optional): Arguments to instantiate the preifx of
            uri corresponding backend. Defaults to None.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
    """
    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 ignore_empty: bool = False,):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.ignore_empty = ignore_empty

    def transform(self, results: dict) -> Optional[dict]:
        """The transform function of :class:`LoadImage`.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        try:
            if 'img' not in results:
                img = cv2.imread(results['img_path'])
                results['img'] = img
            else:
                img = results['img']
                assert isinstance(img, np.ndarray)

            if self.to_float32:
                img = img.astype(np.float32)

            if 'img_path' not in results:
                results['img_path'] = None
            results['img_shape'] = img.shape[:2]
            results['ori_shape'] = img.shape[:2]
        except Exception as e:
            e = type(e)(
                f'`{str(e)}` occurs when loading `{results["img_path"]}`.'
                'Please check whether the file exists.')
            raise e

        return results

    def __call__(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        return self.transform(results)

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', ")

        return repr_str
