# Copyright (c) OpenMMLab. All rights reserved.
import math

import pytest
import torch
from torch import nn

from mmdet3d.models.ir_heads.keypoint_head_attention import KeypointHeadAttention


HEIGHT, WIDTH = 32, 48
CENTER = torch.tensor([9.8, 6.6])


def make_head(**kwargs):
    # Non-square maps and unequal voxel sizes catch x/y or stride mix-ups.
    options = dict(
        in_channels=2,
        d_model=16,
        transformer_nhead=4,
        transformer_num_layers=1,
        transformer_dim_feedforward=32,
        transformer_dropout=0.,
        fc_layers_share=dict(fc_channels=[16], output_channels=16),
        fc_layers_bbox=dict(fc_channels=[16]),
        tasks=[dict(num_class=1) for _ in range(3)],
        bbox_code_size=8,
        train_cfg=dict(
            grid_size=[384, 256, 40],
            out_size_factor=8,
            voxel_size=[0.1, 0.2, 0.2],
            point_cloud_range=[0., -10., -3., 38.4, 41.2, 5.]))
    options.update(kwargs)
    return KeypointHeadAttention(**options).eval()


def coordinate_map():
    # Each feature value is its physical coordinate, so interpolation directly
    # exposes the physical position actually sampled from the feature map.
    ys = torch.arange(HEIGHT) * 1.6 - 10.
    xs = torch.arange(WIDTH) * 0.8
    y, x = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([x, y]).unsqueeze(0)


def proposals(length=4., width=1.6, yaw=0., norm_bbox=True,
              rotation_scale=1., code_size=8):
    box = torch.zeros(1, 1, code_size)
    box[..., :2] = torch.tensor([0.25, 0.375])
    dims = torch.tensor([length, width, 1.5])
    box[..., 3:6] = dims.log() if norm_bbox else dims
    box[..., 6] = math.sin(yaw) * rotation_scale
    box[..., 7] = math.cos(yaw) * rotation_scale
    return box


def center_indices(count=1):
    return torch.full((1, count), 10 * WIDTH + 12, dtype=torch.long)


@pytest.mark.parametrize('norm_bbox', [True, False])
@pytest.mark.parametrize('length,width', [(4., 1.6), (0.6, 0.4)])
@pytest.mark.parametrize('yaw', [0., math.pi / 4, -math.pi / 2])
def test_box_samples_use_physical_dimensions(norm_bbox, length, width, yaw):
    head = make_head(norm_bbox=norm_bbox)
    box = proposals(length, width, yaw, norm_bbox, rotation_scale=3.)
    original = box.clone()
    sampled = head._extract_feat(coordinate_map(), box, center_indices())
    # Independent geometric expectation for center and four edge midpoints.
    forward = torch.tensor([math.cos(yaw), math.sin(yaw)]) * length / 2
    left = torch.tensor([-math.sin(yaw), math.cos(yaw)]) * width / 2
    expected = torch.stack([
        CENTER, CENTER + forward, CENTER - left,
        CENTER - forward, CENTER + left])
    torch.testing.assert_close(sampled[0, 0, :, :2], expected,
                               atol=1e-5, rtol=1e-5)
    assert sampled.shape == (1, 1, 5, 4)
    torch.testing.assert_close(box, original)
    assert head._out_stride == 8


@pytest.mark.parametrize('radius', [0.5, 1., 2.])
def test_context_radius_is_independent_of_box_size_and_yaw(radius):
    head = make_head(context_radius=radius, context_tasks=[1, 2])
    small = proposals(0.6, 0.4, yaw=0.)
    large = proposals(4., 1.6, yaw=math.pi / 4)
    samples = []
    for box in [small, large]:
        result = head._extract_feat(
            coordinate_map(), box, center_indices(), task_id=1)
        assert result.shape == (1, 1, 13, 4)
        context = result[0, 0, 5:, :2]
        distances = torch.linalg.vector_norm(context - CENTER, dim=-1)
        torch.testing.assert_close(distances, torch.full((8,), radius),
                                   atol=1e-5, rtol=1e-5)
        samples.append(context)
    torch.testing.assert_close(samples[0], samples[1])
    # The diagonal also lies on the circle, rather than at sqrt(2) * radius.
    expected_diagonal = CENTER + radius / math.sqrt(2)
    torch.testing.assert_close(samples[0][1], expected_diagonal)
    assert head._extract_feat(coordinate_map(), small, center_indices(),
                              task_id=0).shape[2] == 5
    assert head._extract_feat(coordinate_map(), small, center_indices(),
                              task_id=2).shape[2] == 13


def test_context_can_be_disabled_or_enabled_for_all_tasks():
    box = proposals()
    baseline = make_head()
    assert baseline.positional_encoding.num_embeddings == 5
    assert baseline._extract_feat(coordinate_map(), box,
                                  center_indices()).shape[2] == 5
    contextual = make_head(context_radius=1.)
    for task_id in range(3):
        assert contextual._extract_feat(
            coordinate_map(), box, center_indices(), task_id).shape[2] == 13


@pytest.mark.parametrize('radius', [0., -1., float('nan'), float('inf')])
def test_invalid_context_radius(radius):
    with pytest.raises(ValueError, match='context_radius'):
        make_head(context_radius=radius)


def test_samples_at_map_boundary_are_finite():
    head = make_head(context_radius=2.)
    box = proposals()
    box[..., :2] = 0.
    sampled = head._extract_feat(coordinate_map(), box,
                                 torch.zeros(1, 1, dtype=torch.long))
    assert torch.isfinite(sampled).all()
    assert (sampled[..., 0] >= 0).all()
    assert (sampled[..., 1] >= -10.).all()


@pytest.mark.parametrize('code_size', [8, 10])
def test_zero_residual_and_context_gradients(code_size):
    head = make_head(bbox_code_size=code_size, context_radius=1.,
                     context_tasks=[1, 2])
    box = proposals(code_size=code_size).expand(1, 2, -1).clone()
    feature_map = coordinate_map().requires_grad_()
    scores = torch.ones(1, 2, 1)
    for task_id in range(3):
        out = head(feature_map, box, center_indices(2), task_id, scores)
        torch.testing.assert_close(out, box)
    # After the zero-initialized residual projection starts learning, gradients
    # must reach both the sampled feature map and the new context embeddings.
    nn.init.normal_(head.fc_layers_bbox[1][-1].weight, std=0.01)
    out = head(feature_map, box, center_indices(2), task_id=1, scores=scores)
    out.square().sum().backward()
    assert torch.isfinite(out).all()
    assert feature_map.grad is not None
    assert torch.isfinite(feature_map.grad).all()
    assert feature_map.grad.abs().sum() > 0
    context_grad = head.positional_encoding.weight.grad[5:]
    assert torch.isfinite(context_grad).all()
    assert context_grad.abs().sum() > 0


def test_unrefined_task_returns_proposals():
    head = make_head(context_radius=1., refine_tasks=[0])
    box = proposals()
    assert head(coordinate_map(), box, center_indices(), task_id=1) is box


@pytest.mark.parametrize('suffix', ['', '-minified'])
def test_kitti_context_targets_match_dataset_classes(suffix):
    from pathlib import Path
    from mmengine.config import Config

    root = Path(__file__).resolve().parents[3]
    config = root / 'configs/centerpoint' / (
        'ir_centerpoint_voxel01_second_secfpn_8xb4-cyclic-80e'
        f'_kitti-3d{suffix}.py')
    cfg = Config.fromfile(str(config))
    dataset_cfg = cfg.train_dataloader.dataset
    if dataset_cfg.type == 'RepeatDataset':
        dataset_cfg = dataset_cfg.dataset
    classes = list(dataset_cfg.metainfo.classes)
    head = cfg.model.pts_bbox_head
    task_classes = [name for task in head.tasks for name in task.class_names]
    assert task_classes == classes
    assert [classes[i] for i in head.ir_head.context_tasks] == [
        'Pedestrian', 'Cyclist']
