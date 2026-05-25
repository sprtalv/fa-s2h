from fas2h.route.cache_keys import build_cache_key


def test_cache_key_is_stable_for_same_input() -> None:
    key1 = build_cache_key(
        pair_id="toy_0001",
        model_name="openclip_vit_b32",
        pretrained="laion2b_s34b_b79k",
        layers=[2, 3, 4],
        topk_ratio=0.1,
        alpha_attn=1 / 3,
        beta_cls=1 / 3,
        gamma_global=1 / 3,
        lambda_path=0.5,
        projection_type="identity",
        source_path="tests/assets/smoke_source.png",
        target_path="tests/assets/smoke_target.png",
    )
    key2 = build_cache_key(
        pair_id="toy_0001",
        model_name="openclip_vit_b32",
        pretrained="laion2b_s34b_b79k",
        layers=[2, 3, 4],
        topk_ratio=0.1,
        alpha_attn=1 / 3,
        beta_cls=1 / 3,
        gamma_global=1 / 3,
        lambda_path=0.5,
        projection_type="identity",
        source_path="tests/assets/smoke_source.png",
        target_path="tests/assets/smoke_target.png",
    )
    assert key1 == key2


def test_cache_key_changes_when_layers_or_topk_change() -> None:
    key_a = build_cache_key(
        pair_id="toy_0001",
        model_name="openclip_vit_b32",
        pretrained="laion2b_s34b_b79k",
        layers=[2, 3, 4],
        topk_ratio=0.1,
        alpha_attn=1 / 3,
        beta_cls=1 / 3,
        gamma_global=1 / 3,
        lambda_path=0.5,
        projection_type="identity",
        source_path="tests/assets/smoke_source.png",
        target_path="tests/assets/smoke_target.png",
    )
    key_b = build_cache_key(
        pair_id="toy_0001",
        model_name="openclip_vit_b32",
        pretrained="laion2b_s34b_b79k",
        layers=[2, 3, 5],
        topk_ratio=0.1,
        alpha_attn=1 / 3,
        beta_cls=1 / 3,
        gamma_global=1 / 3,
        lambda_path=0.5,
        projection_type="identity",
        source_path="tests/assets/smoke_source.png",
        target_path="tests/assets/smoke_target.png",
    )
    key_c = build_cache_key(
        pair_id="toy_0001",
        model_name="openclip_vit_l14",
        pretrained="laion2b_s32b_b82k",
        layers=[2, 3, 4],
        topk_ratio=0.2,
        alpha_attn=1 / 3,
        beta_cls=1 / 3,
        gamma_global=1 / 3,
        lambda_path=0.5,
        projection_type="identity",
        source_path="tests/assets/smoke_source.png",
        target_path="tests/assets/smoke_target.png",
    )
    assert key_a != key_b
    assert key_a != key_c
