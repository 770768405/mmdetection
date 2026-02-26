# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for HolographicDataset."""

import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pytest

from mmdet.datasets import DATASETS, HolographicDataset


class TestHolographicDataset(unittest.TestCase):
    """Test cases for HolographicDataset."""

    @patch('mmdet.datasets.HolographicDataset.load_annotations')
    @patch('mmdet.datasets.HolographicDataset._filter_imgs')
    def test_holographic_dataset_registration(self, mock_filter, mock_load):
        """Test that HolographicDataset is properly registered."""
        # Verify dataset is registered
        assert 'HolographicDataset' in DATASETS.module_dict
        
        # Get the dataset class
        dataset_class = DATASETS.get('HolographicDataset')
        assert dataset_class == HolographicDataset

    @patch('mmdet.datasets.HolographicDataset.load_annotations')
    @patch('mmdet.datasets.HolographicDataset._filter_imgs')
    def test_holographic_dataset_initialization(self, mock_filter, mock_load):
        """Test HolographicDataset can be initialized."""
        mock_load.return_value = []
        
        # Test basic initialization
        dataset = HolographicDataset(
            ann_file='test.json',
            pipeline=[],
            test_mode=True)
        
        assert dataset is not None
        assert dataset.depth_prefix is None
        assert dataset.phase_prefix is None

    @patch('mmdet.datasets.HolographicDataset.load_annotations')
    @patch('mmdet.datasets.HolographicDataset._filter_imgs')
    def test_holographic_dataset_with_prefixes(self, mock_filter, mock_load):
        """Test HolographicDataset with depth and phase prefixes."""
        mock_load.return_value = []
        
        # Test with depth and phase prefixes
        dataset = HolographicDataset(
            ann_file='test.json',
            pipeline=[],
            depth_prefix='depth/',
            phase_prefix='phase/',
            test_mode=True)
        
        assert dataset.depth_prefix == 'depth/'
        assert dataset.phase_prefix == 'phase/'

    @patch('mmdet.datasets.HolographicDataset.load_annotations')
    @patch('mmdet.datasets.HolographicDataset._filter_imgs')
    def test_pre_pipeline(self, mock_filter, mock_load):
        """Test pre_pipeline adds holographic fields."""
        mock_load.return_value = []
        
        dataset = HolographicDataset(
            ann_file='test.json',
            pipeline=[],
            depth_prefix='depth/',
            phase_prefix='phase/',
            test_mode=True)
        
        results = {'img_prefix': 'images/'}
        dataset.pre_pipeline(results)
        
        # Check holographic fields are added
        assert 'depth_prefix' in results
        assert 'phase_prefix' in results
        assert 'is_holographic' in results
        assert results['is_holographic'] is True
        assert results['depth_prefix'] == 'depth/'
        assert results['phase_prefix'] == 'phase/'

    @patch('mmdet.datasets.HolographicDataset.load_annotations')
    @patch('mmdet.datasets.HolographicDataset._filter_imgs')
    def test_custom_classes(self, mock_filter, mock_load):
        """Test HolographicDataset with custom classes."""
        mock_load.return_value = []
        
        # Test with custom classes as tuple
        dataset = HolographicDataset(
            ann_file='test.json',
            pipeline=[],
            classes=('class1', 'class2'),
            test_mode=True)
        
        assert dataset.CLASSES == ('class1', 'class2')
        
        # Test with custom classes as list
        dataset = HolographicDataset(
            ann_file='test.json',
            pipeline=[],
            classes=['class1', 'class2', 'class3'],
            test_mode=True)
        
        assert dataset.CLASSES == ['class1', 'class2', 'class3']


@pytest.mark.parametrize('load_depth,load_phase', [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
])
def test_holographic_pipeline(load_depth, load_phase):
    """Test that holographic pipeline configuration is valid."""
    from mmdet.datasets.pipelines import LoadHolographicImage
    
    # Test pipeline component initialization
    loader = LoadHolographicImage(
        load_depth=load_depth,
        load_phase=load_phase)
    
    assert loader.load_depth == load_depth
    assert loader.load_phase == load_phase
    assert loader.to_float32 is False
    
    # Test repr
    repr_str = repr(loader)
    assert 'LoadHolographicImage' in repr_str
    assert f'load_depth={load_depth}' in repr_str
    assert f'load_phase={load_phase}' in repr_str


if __name__ == '__main__':
    unittest.main()
