import torch

def chamfer_distance(pc1, pc2):
    """
    pc1, pc2: B x N x 3
    """
    diff = pc1.unsqueeze(2) - pc2.unsqueeze(1)  # B x N x N x 3
    dist = torch.sum(diff ** 2, dim=-1)         # B x N x N
    cd1 = torch.mean(torch.min(dist, dim=2)[0], dim=1)
    cd2 = torch.mean(torch.min(dist, dim=1)[0], dim=1)
    return torch.mean(cd1 + cd2)
