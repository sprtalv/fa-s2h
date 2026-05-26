import torch

from fas2h.features import FeatureBundle
from fas2h.route.projection import IdentityProjection
from fas2h.route.source_carrier import select_source_carriers


def make_feature_bundle(num_patches: int = 16, dim: int = 8) -> FeatureBundle:
    batch = 1
    heads = 2
    patch_tokens = torch.randn(batch, num_patches, dim)
    cls_tokens = torch.randn(batch, dim)
    attentions = torch.softmax(torch.randn(batch, heads, num_patches + 1, num_patches + 1), dim=-1)
    return FeatureBundle(
        patch_tokens_by_layer={0: patch_tokens},
        cls_tokens_by_layer={0: cls_tokens},
        attentions_by_layer={0: attentions},
        final_global=torch.randn(batch, dim),
        num_layers=1,
        num_patches=num_patches,
    )


def test_random_selection_same_seed_same_indices() -> None:
    fb = make_feature_bundle()
    score_cfg = {"alpha_attn": 1 / 3, "beta_cls": 1 / 3, "gamma_global": 1 / 3}
    first = select_source_carriers(
        feature_bundle=fb,
        model_name="m",
        pretrained="p",
        layers=[0],
        topk_ratio=0.2,
        projection=IdentityProjection(),
        score_cfg=score_cfg,
        selection_mode="random_patch",
        random_seed=42,
        pair_id="pair_a",
    )[0].source_indices
    second = select_source_carriers(
        feature_bundle=fb,
        model_name="m",
        pretrained="p",
        layers=[0],
        topk_ratio=0.2,
        projection=IdentityProjection(),
        score_cfg=score_cfg,
        selection_mode="random_patch",
        random_seed=42,
        pair_id="pair_a",
    )[0].source_indices
    assert torch.equal(first, second)


def test_random_selection_different_seed_different_indices() -> None:
    fb = make_feature_bundle()
    score_cfg = {"alpha_attn": 1 / 3, "beta_cls": 1 / 3, "gamma_global": 1 / 3}
    first = select_source_carriers(
        feature_bundle=fb,
        model_name="m",
        pretrained="p",
        layers=[0],
        topk_ratio=0.2,
        projection=IdentityProjection(),
        score_cfg=score_cfg,
        selection_mode="random_patch",
        random_seed=42,
        pair_id="pair_a",
    )[0].source_indices
    second = select_source_carriers(
        feature_bundle=fb,
        model_name="m",
        pretrained="p",
        layers=[0],
        topk_ratio=0.2,
        projection=IdentityProjection(),
        score_cfg=score_cfg,
        selection_mode="random_patch",
        random_seed=43,
        pair_id="pair_a",
    )[0].source_indices
    assert not torch.equal(first, second)


def test_random_selection_index_range_and_count() -> None:
    num_patches = 25
    topk_ratio = 0.2
    fb = make_feature_bundle(num_patches=num_patches)
    score_cfg = {"alpha_attn": 1 / 3, "beta_cls": 1 / 3, "gamma_global": 1 / 3}
    bundle = select_source_carriers(
        feature_bundle=fb,
        model_name="m",
        pretrained="p",
        layers=[0],
        topk_ratio=topk_ratio,
        projection=IdentityProjection(),
        score_cfg=score_cfg,
        selection_mode="random_patch",
        random_seed=42,
        pair_id="pair_a",
    )[0]
    assert bundle.source_indices is not None
    topk = int(num_patches * topk_ratio)
    assert bundle.source_indices.shape[-1] == topk
    assert int(bundle.source_indices.min()) >= 0
    assert int(bundle.source_indices.max()) < num_patches
