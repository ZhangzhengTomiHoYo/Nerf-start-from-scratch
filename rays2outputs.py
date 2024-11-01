import torch
import torch.nn.functional as F
device = 'cuda'
def raws2outputs(raw,
                 z_vals,
                 rays_d,
                 raw_noise_std=0,
                 white_bkgd=False):
    """将模型预测结果转换为有语义价值的值，如RGB图、深度图等
    Args:
    raw: [mum_rays, num - samples(射线上的样本数) ，4].模型的预测结果
    z_vals: [num_rays, num_samples(射线上的样本数)].样本的间隔
    rays_d: [num_rays, 3].射线方向
    Returns:
    rgb_map: [num_rays, 3] • 预测的射线渲染的RGB颜色
    disp_map: [num_rays].预测的视差图(深度图求逆运算)
    acc.map: [num_rays].射线方向权重的和
    weights: [numjays, num.samples].每个采样颜色上的权重
    depth.map: [num.rays].预测得到的深度
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu : 1. - torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(device)], -1)

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])

    noise = 0
    if raw_noise_std > 0:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

    alpha = raw2alpha(raw[...,3] + noise, dists)

    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device),1. - alpha + 1e-10],-1),-1)[:,-1]

    rgb_map = torch.sum(weights[...,None] * rgb, -2)

    depth_map = torch.sum(weights * z_vals, -1)

    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))

    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map