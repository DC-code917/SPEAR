import torch


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state)
    return model
