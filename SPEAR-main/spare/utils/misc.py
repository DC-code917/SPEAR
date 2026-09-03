import torch


def count_lines(file_path):
    lines_num = 0
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(2 ** 20)
            if not data:
                break
            lines_num += data.count(b'\n')
    return lines_num


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def pooling(memory_bank, attention_mask, pooling_type="mean"):
    mask = attention_mask.bool()
    expanded = mask.unsqueeze(-1).type_as(memory_bank)
    if pooling_type == "mean":
        features = torch.sum(memory_bank * expanded, dim=1)
        features = torch.div(features, torch.sum(expanded, dim=1).clamp_min(1.0))
    elif pooling_type == "last":
        indices = mask.sum(dim=1).clamp_min(1) - 1
        features = memory_bank[torch.arange(memory_bank.shape[0], device=memory_bank.device), indices]
    elif pooling_type == "max":
        features = memory_bank.masked_fill(~mask.unsqueeze(-1), torch.finfo(memory_bank.dtype).min).max(dim=1).values
    else:
        features = memory_bank[:, 0, :]
    return features
