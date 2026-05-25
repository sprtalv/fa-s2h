from fas2h.viz.patch_grid import infer_patch_grid


def test_patch_grid_for_196() -> None:
    assert infer_patch_grid(196) == (14, 14)
