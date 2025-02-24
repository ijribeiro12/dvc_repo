##############################
# SSIM for 1D signals
##############################
def ssim_1d(pred, target, C1=0.01**2, C2=0.03**2):
    """
    A simplified, global SSIM for 1D signals:
    pred, target: shape (batch, T, 1) or (batch, T)
    """
    if pred.ndim == 3 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)  # => (batch, T)
    if target.ndim == 3 and target.size(-1) == 1:
        target = target.squeeze(-1)

    mu_x = pred.mean(dim=1, keepdim=True)
    mu_y = target.mean(dim=1, keepdim=True)
    sigma_x = ((pred - mu_x)**2).mean(dim=1, keepdim=True)
    sigma_y = ((target - mu_y)**2).mean(dim=1, keepdim=True)
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=1, keepdim=True)
    
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = numerator / denominator
    return ssim_map.mean()  # average over batch