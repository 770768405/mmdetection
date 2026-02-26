# Copyright (c) OpenMMLab. All rights reserved.
"""Holographic imaging dataset for object detection.

This dataset is designed to handle holographic images which may contain
additional depth or phase information beyond standard RGB images.
Holographic imaging is commonly used in microscopy and biomedical applications.
"""

import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class HolographicDataset(CustomDataset):
    """Holographic imaging dataset for detection.

    This dataset extends CustomDataset to support holographic images,
    which may include:
    - Multi-plane reconstructions
    - Phase and amplitude information
    - Depth-resolved imaging data

    The annotation format follows the CustomDataset format with optional
    holographic-specific metadata.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        classes (str | Sequence[str], optional): Specify classes to load.
        data_root (str, optional): Data root for file paths.
        img_prefix (str, optional): Prefix for image paths.
        depth_prefix (str, optional): Prefix for depth map paths if available.
        phase_prefix (str, optional): Prefix for phase map paths if available.
        test_mode (bool, optional): If True, annotations will not be loaded.
        filter_empty_gt (bool, optional): If True, filter images without boxes.
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 depth_prefix=None,
                 phase_prefix=None,
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk')):
        self.depth_prefix = depth_prefix
        self.phase_prefix = phase_prefix
        
        super(HolographicDataset, self).__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            data_root=data_root,
            img_prefix=img_prefix,
            seg_prefix=seg_prefix,
            proposal_file=proposal_file,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
            file_client_args=file_client_args)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline.

        Extends parent method to add holographic-specific fields.

        Args:
            results (dict): Result dict.
        """
        super(HolographicDataset, self).pre_pipeline(results)
        
        # Add holographic-specific prefixes
        if self.depth_prefix is not None:
            results['depth_prefix'] = self.depth_prefix
        if self.phase_prefix is not None:
            results['phase_prefix'] = self.phase_prefix
            
        # Add flag indicating holographic data
        results['is_holographic'] = True

    def load_annotations(self, ann_file):
        """Load annotation from annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from the annotation file.
        """
        # Load annotations using parent method
        data_infos = super(HolographicDataset, self).load_annotations(ann_file)
        
        # Optionally process holographic-specific metadata
        # This could include depth information, reconstruction parameters, etc.
        for data_info in data_infos:
            # Add holographic metadata if present in annotations
            if 'holographic_metadata' not in data_info:
                data_info['holographic_metadata'] = {}
                
        return data_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        ann_info = super(HolographicDataset, self).get_ann_info(idx)
        
        # Add holographic metadata to annotation info
        data_info = self.data_infos[idx]
        if 'holographic_metadata' in data_info:
            ann_info['holographic_metadata'] = data_info['holographic_metadata']
            
        return ann_info
