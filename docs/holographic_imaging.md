# Holographic Imaging Technology Support

This feature adds support for holographic imaging datasets in MMDetection, enabling object detection on holographic images commonly used in microscopy and biomedical applications.

## Overview

Holographic imaging captures both amplitude and phase information of light waves, allowing for depth-resolved and 3D reconstruction of objects. This implementation provides:

1. **HolographicDataset**: A dataset class that extends CustomDataset to handle holographic images
2. **LoadHolographicImage**: A pipeline component for loading holographic data including optional depth and phase maps
3. **Configuration templates**: Ready-to-use configuration files for holographic detection tasks

## Features

### HolographicDataset

The `HolographicDataset` class inherits from `CustomDataset` and adds support for:
- Multi-modal data (intensity, depth, phase)
- Holographic-specific metadata
- Custom preprocessing pipelines for holographic data

**Key Parameters:**
- `depth_prefix`: Path prefix for depth map files (optional)
- `phase_prefix`: Path prefix for phase map files (optional)
- All standard CustomDataset parameters

### LoadHolographicImage Pipeline

The `LoadHolographicImage` pipeline component extends standard image loading with:
- Optional depth map loading (`load_depth=True`)
- Optional phase map loading (`load_phase=True`)
- Automatic fallback if depth/phase data is unavailable
- Consistent interface with other MMDetection loaders

## Usage

### Basic Configuration

```python
# Use the holographic dataset configuration
_base_ = './configs/_base_/datasets/holographic_detection.py'

# Or define custom configuration
dataset_type = 'HolographicDataset'
data_root = 'data/holographic/'

train_pipeline = [
    dict(
        type='LoadHolographicImage',
        load_depth=True,  # Enable depth map loading
        load_phase=False  # Disable phase map loading
    ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'train/',
        depth_prefix=data_root + 'train_depth/',  # Optional
        phase_prefix=data_root + 'train_phase/',  # Optional
        pipeline=train_pipeline
    ),
    # ... val and test configs
)
```

### Data Format

The holographic dataset expects:

1. **Annotation file**: Standard COCO or CustomDataset JSON format
2. **Image files**: Holographic intensity images (standard formats: PNG, JPEG, TIFF)
3. **Depth maps** (optional): Grayscale images with depth information
4. **Phase maps** (optional): Grayscale images with phase information

Directory structure:
```
data/holographic/
├── annotations/
│   ├── train.json
│   └── val.json
├── train/
│   ├── image1.png
│   └── image2.png
├── train_depth/  # Optional
│   ├── image1.png
│   └── image2.png
└── train_phase/  # Optional
    ├── image1.png
    └── image2.png
```

## Example: Training with Holographic Data

```bash
# Using the provided configuration
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_holographic.py

# With custom config inheriting holographic base
python tools/train.py configs/your_holographic_config.py
```

## Testing

Unit tests are provided in `tests/test_data/test_datasets/test_holographic_dataset.py`:

```bash
pytest tests/test_data/test_datasets/test_holographic_dataset.py -v
```

## Implementation Details

### Class Hierarchy

```
Dataset (PyTorch)
    └── CustomDataset (MMDetection)
        └── HolographicDataset (New)
```

### Pipeline Components

```
LoadImageFromFile (Standard)
    └── LoadHolographicImage (New)
        ├── Loads main intensity image
        ├── Optionally loads depth map
        └── Optionally loads phase map
```

## Applications

This implementation is suitable for:
- Digital holographic microscopy (DHM)
- Particle tracking in holographic videos
- Cell detection and segmentation
- Biomedical imaging with holographic data
- Any detection task requiring depth or phase information

## Future Enhancements

Potential improvements for future versions:
- Multi-plane reconstruction support
- Phase unwrapping integration
- Holographic-specific augmentation techniques
- 3D bounding box support for depth-resolved detection
- Integration with holographic reconstruction algorithms

## References

- [OpenMMLab Detection Toolbox](https://github.com/open-mmlab/mmdetection)
- Digital Holographic Microscopy applications
- Phase imaging techniques in microscopy

## Citation

If you use this holographic imaging support in your research, please cite:

```bibtex
@misc{mmdet-holographic,
  title={Holographic Imaging Support for MMDetection},
  author={MMDetection Contributors},
  year={2026},
  howpublished={\url{https://github.com/770768405/mmdetection}}
}
```
