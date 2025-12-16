import torch

def add_dp_noise(model, max_norm, noise_multiplier, batch_size):
    """
    Gradient clipping + Gaussian noise (DP-SGD).
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return

    # Clip
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    clip = max_norm / (total_norm + 1e-6)
    clip = min(clip, 1.0)
    for g in grads:
        g.mul_(clip)

    # Noise
    std = noise_multiplier * max_norm / (batch_size ** 0.5)
    for g in grads:
        g.add_(torch.randn_like(g) * std)
