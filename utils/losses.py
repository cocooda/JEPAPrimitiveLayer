import torch
import torch.nn.functional as F

def jepa_masked_mse(predicted, target, mask):
    B, N, D = predicted.shape
    mask = mask.unsqueeze(-1).expand(-1, -1, D)
    diff = (predicted - target)**2
    masked_diff = diff[mask].view(B, -1) if mask.any() else diff.new_zeros((B, 0))
    if masked_diff.numel() == 0:
        return torch.tensor(0.0, device=predicted.device)
    return masked_diff.mean()

def vicreg_loss(z1, z2, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0):
    sim_loss = F.mse_loss(z1, z2)
    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))
    z1_centered = z1 - z1.mean(dim=0)
    z2_centered = z2 - z2.mean(dim=0)
    N, D = z1.shape
    cov_z1 = (z1_centered.T @ z1_centered) / (N - 1)
    cov_z2 = (z2_centered.T @ z2_centered) / (N - 1)
    cov_z1.fill_diagonal_(0)
    cov_z2.fill_diagonal_(0)
    cov_loss = cov_z1.pow(2).sum() / D + cov_z2.pow(2).sum() / D
    return sim_coeff * sim_loss + var_coeff * var_loss + cov_coeff * cov_loss

def drift_loss(z_c, prev_z_c):
    if prev_z_c is None:
        return torch.tensor(0.0, device=z_c.device)
    return F.mse_loss(z_c, prev_z_c)
