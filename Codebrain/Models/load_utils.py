import torch


def try_load_backbone_weights(
    backbone,
    foundation_dir,
    cuda_idx=None,
    map_location=None,
    min_loaded_ratio=0.0,
):
    try:
        if map_location is None:
            if cuda_idx is None:
                map_location = "cpu"
            else:
                map_location = torch.device(f"cuda:{cuda_idx}")
        state = torch.load(foundation_dir, map_location=map_location)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if not isinstance(state, dict):
            print(f"[warn] Unsupported checkpoint format: {type(state)}")
            return False

        cleaned = {}
        for key, value in state.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("backbone."):
                key = key[len("backbone.") :]
            cleaned[key] = value

        model_state = backbone.state_dict()
        matched = {}
        for key, value in cleaned.items():
            if key in model_state and model_state[key].shape == value.shape:
                matched[key] = value

        total_model_keys = max(len(model_state), 1)
        loaded = len(matched)
        loaded_ratio = loaded / total_model_keys
        if loaded_ratio < float(min_loaded_ratio):
            print(
                '[warn] Skip loading mismatched backbone weights - '
                f'loaded={loaded}/{len(model_state)} ({loaded_ratio:.3f}) '
                f'< min_loaded_ratio={float(min_loaded_ratio):.3f}'
            )
            return False

        missing, unexpected = backbone.load_state_dict(matched, strict=False)
        print(
            "[info] Backbone load result - "
            f"loaded={loaded}, missing={len(missing)}, unexpected={len(unexpected)}, "
            f"loaded_ratio={loaded_ratio:.3f}"
        )
        return loaded > 0
    except Exception as exc:
        print(f"[warn] Failed to load backbone weights from {foundation_dir}: {exc}")
        return False
