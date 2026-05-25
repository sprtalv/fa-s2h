import torch

from fas2h.route.rollout import (
    add_residual_and_normalize,
    average_heads,
    compute_rollout,
    extract_cls_to_patch_rollout,
)


def test_rollout_helpers_shapes_and_normalization() -> None:
    attn = torch.tensor(
        [
            [
                [
                    [0.5, 0.2, 0.2, 0.1],
                    [0.1, 0.7, 0.1, 0.1],
                    [0.2, 0.2, 0.5, 0.1],
                    [0.1, 0.2, 0.2, 0.5],
                ],
                [
                    [0.4, 0.3, 0.2, 0.1],
                    [0.1, 0.6, 0.2, 0.1],
                    [0.1, 0.2, 0.6, 0.1],
                    [0.2, 0.2, 0.1, 0.5],
                ],
            ]
        ],
        dtype=torch.float32,
    )
    avg = average_heads(attn)
    assert avg.shape == (1, 4, 4)

    norm = add_residual_and_normalize(avg)
    assert norm.shape == (1, 4, 4)
    row_sums = norm.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_extract_cls_to_patch_rollout_matches_num_patches() -> None:
    attn_by_layer = {
        1: torch.rand(1, 2, 5, 5),
        2: torch.rand(1, 2, 5, 5),
        3: torch.rand(1, 2, 5, 5),
    }
    rollout = compute_rollout(attn_by_layer, start_layer=1, end_layer=3)
    cls_to_patch = extract_cls_to_patch_rollout(rollout)
    assert rollout.shape == (1, 5, 5)
    assert cls_to_patch.shape == (1, 4)
