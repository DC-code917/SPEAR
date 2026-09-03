import torch


def save_model(model, model_path):
    target = model.module if hasattr(model, "module") else model
    payload = {"model": target.state_dict()}
    if hasattr(target, "config"):
        payload["config"] = target.config.to_dict()
    torch.save(payload, model_path)
