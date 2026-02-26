# Copyright (c) OpenMMLab. All rights reserved.
"""Pipeline components for loading holographic imaging data."""

import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadHolographicImage:
    """Load holographic images with optional depth and phase information.

    This pipeline component extends standard image loading to support
    holographic imaging data, which may include:
    - Standard RGB/grayscale intensity images
    - Depth maps from holographic reconstruction
    - Phase maps containing phase information

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Optional keys are "depth_prefix" and "phase_prefix".
    
    Added or updated keys are:
    - "filename", "img", "img_shape", "ori_shape", "pad_shape", "scale_factor"
    - "depth_map" (optional), "phase_map" (optional)
    - "holographic_fields" (list of additional holographic data fields)

    Args:
        to_float32 (bool): Whether to convert images to float32.
            Defaults to False.
        color_type (str): Flag for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        load_depth (bool): Whether to load depth maps if available.
            Defaults to False.
        load_phase (bool): Whether to load phase maps if available.
            Defaults to False.
        file_client_args (dict): Arguments for FileClient.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr',
                 load_depth=False,
                 load_phase=False,
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.load_depth = load_depth
        self.load_phase = load_phase
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_image(self, filename):
        """Load an image file.

        Args:
            filename (str): Path to image file.

        Returns:
            np.ndarray: Loaded image.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        
        if self.to_float32:
            img = img.astype(np.float32)
            
        return img

    def __call__(self, results):
        """Load holographic image and optional depth/phase data.

        Args:
            results (dict): Result dict from dataset.

        Returns:
            dict: Updated result dict with loaded data.
        """
        # Load main image
        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                               results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img = self._load_image(filename)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        
        # Track holographic-specific fields
        holographic_fields = []

        # Load depth map if requested and available
        if self.load_depth and results.get('depth_prefix') is not None:
            depth_filename = osp.join(
                results['depth_prefix'],
                results['img_info']['filename'])
            
            try:
                # Load depth as grayscale
                depth_bytes = self.file_client.get(depth_filename)
                depth_map = mmcv.imfrombytes(depth_bytes, flag='grayscale')
                
                if self.to_float32:
                    depth_map = depth_map.astype(np.float32)
                    
                results['depth_map'] = depth_map
                holographic_fields.append('depth_map')
            except (FileNotFoundError, IOError):
                # Depth map not available, skip
                pass

        # Load phase map if requested and available  
        if self.load_phase and results.get('phase_prefix') is not None:
            phase_filename = osp.join(
                results['phase_prefix'],
                results['img_info']['filename'])
            
            try:
                # Load phase as grayscale
                phase_bytes = self.file_client.get(phase_filename)
                phase_map = mmcv.imfrombytes(phase_bytes, flag='grayscale')
                
                if self.to_float32:
                    phase_map = phase_map.astype(np.float32)
                    
                results['phase_map'] = phase_map
                holographic_fields.append('phase_map')
            except (FileNotFoundError, IOError):
                # Phase map not available, skip
                pass

        results['holographic_fields'] = holographic_fields
        
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"channel_order='{self.channel_order}', "
                    f'load_depth={self.load_depth}, '
                    f'load_phase={self.load_phase}, '
                    f'file_client_args={self.file_client_args})')
        return repr_str
