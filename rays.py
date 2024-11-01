from logging import DEBUG

import torch
device = 'cuda'
def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                lindisp=False,
                perturb=0,
                network_fine=None,
                raw_noise_std=0,
                white_bkgd=True):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6]
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8],[-1,1,2])
    near, far = bounds[...,0], bounds[...,1]

    t_vals = torch.linspace(0.,1., steps=N_samples).to(device)
    if not lindisp:
        z_vals = near * (1 - t_vals) + far * (t_vals)
    else:
        pass

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0 :
        pass

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    raw = network_query_fn(pts, viewdirs, network_fine)

    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, noise_std, white_bkgd)

    if N_importance > 0:
        pass

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    ret['raw'] = raw

    if N_importance > 0:
        pass

    for k in ret:
        if(torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! ")

    return ret