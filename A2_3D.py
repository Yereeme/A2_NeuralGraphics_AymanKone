import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.checkpoint import checkpoint
import math
import time

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()
here = os.path.dirname(os.path.abspath(__file__))
scene_dir = os.path.join(here, "spheres", "spheres")

# ---------- P6: 3D Gaussians and projection ----------

def quaternion_to_rotation(q):
    # q: (N, 4) as (w, x, y, z)
    q = q / q.norm(dim=-1, keepdim=True)      # normalize so it has length 1
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    row0 = torch.stack([r00, r01, r02], dim=-1)    # (N,3)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    R = torch.stack([row0, row1, row2], dim=-2)    # (N,3,3)
    return R

def covariance_3d(scale, quat):
    # scale: (N, 3) positive, quat: (N, 4)
    R = quaternion_to_rotation(quat)          # (N,3,3)
    S = torch.diag_embed(scale)               # (N,3,3) sizes on the diagonal
    M = R @ S
    Sigma = M @ M.transpose(-1, -2)           # R S S^T R^T
    return Sigma

def project_mean(mu_cam, K):
    # mu_cam: (N, 3) points in camera space -> (N, 2) pixel positions
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    xc = mu_cam[:, 0]
    yc = mu_cam[:, 1]
    zc = mu_cam[:, 2]
    u = fx * xc / zc + cx
    v = fy * yc / zc + cy
    mu2 = torch.stack([u, v], dim=-1)         # (N,2)
    return mu2

def projection_jacobian(mu_cam, K):
    # mu_cam: (N, 3) -> J: (N, 2, 3)
    fx = K[0][0]
    fy = K[1][1]
    xc = mu_cam[:, 0]
    yc = mu_cam[:, 1]
    zc = mu_cam[:, 2]
    zeros = torch.zeros_like(zc)
    row0 = torch.stack([fx / zc, zeros, -fx * xc / zc ** 2], dim=-1)    # how u changes
    row1 = torch.stack([zeros, fy / zc, -fy * yc / zc ** 2], dim=-1)    # how v changes
    J = torch.stack([row0, row1], dim=-2)                               # (N,2,3)
    return J

def project_gaussian(mu3, Sigma3, R_wc, t, K):
    # mu3: (N, 3) world means,  Sigma3: (N, 3, 3) world covariances
    mu_cam = mu3 @ R_wc.T + t                  # world -> camera
    mu2 = project_mean(mu_cam, K)              # (N, 2)
    J = projection_jacobian(mu_cam, K)         # (N, 2, 3)
    Scam = R_wc @ Sigma3 @ R_wc.T              # (N, 3, 3)
    Sig2 = J @ Scam @ J.transpose(-1, -2)      # (N, 2, 2)
    depth = mu_cam[:, 2]
    return mu2, Sig2, depth

# ---------- loading the scene ----------

def load_frames(frame_list):
    Rs = []
    ts = []
    images = []
    for frame in frame_list:
        R_wc = torch.tensor(frame["R_wc"], dtype=torch.float32, device=device)
        t = torch.tensor(frame["t"], dtype=torch.float32, device=device)
        image = Image.open(os.path.join(scene_dir, frame["file"])).convert("RGB")
        image = torch.from_numpy(np.array(image)).float().to(device) / 255.0    # 0..1
        Rs.append(R_wc)
        ts.append(t)
        images.append(image)
    return torch.stack(Rs), torch.stack(ts), torch.stack(images)   # (F,3,3), (F,3), (F,H,W,3)

def load_scene():
    with open(os.path.join(scene_dir, "cameras.json")) as f:
        cams = json.load(f)
    K = torch.tensor(cams["K"], dtype=torch.float32, device=device)
    train = load_frames(cams["frames"])
    val = load_frames(cams["val_frames"])
    return K, train, val, cams["convention"]

# ---------- the 2D renderer from A2_main (copied) ----------

def gaussian_weight(xy, mu, Sigma):
    # xy: (P,2) pixels, mu: (N,2) centers, Sigma: (N,2,2) -> (P,N)
    d = xy[:, None, :] - mu[None, :, :]       # (P,N,2)
    Sigma_inv = torch.linalg.inv(Sigma)       # (N,2,2)
    dx = d[..., 0]                            # (P,N)
    dy = d[..., 1]                            # (P,N)
    a = Sigma_inv[:, 0, 0]                    # (N,)
    b = Sigma_inv[:, 0, 1]                    # (N,)
    c = Sigma_inv[:, 1, 1]                    # (N,)
    m = a * dx * dx + 2 * b * dx * dy + c * dy * dy   # (P,N)
    m = m.clamp(min=0)                        # m can never be negative, so w stays <= 1
    return torch.exp(-0.5 * m)

def pixel_grid(H, W):
    ys = torch.arange(H, device=device).float()
    xs = torch.arange(W, device=device).float()
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    xy = torch.stack([grid_x, grid_y], dim=-1)    # (H,W,2)
    return xy.reshape(H * W, 2)                   # (H*W, 2)

def composite(xy, mu, Sigma, color, opacity, order):
    # xy: (P, 2) pixels to draw  ->  (P, 3) colors
    w = gaussian_weight(xy, mu, Sigma)        # (P, N)
    alpha = opacity[None, :] * w              # (P, N)
    alpha = alpha.clamp(max=0.99)             # no puff is ever fully opaque

    alpha = alpha[:, order]                   # put the puffs in front-to-back order
    color = color[order]

    left_after = torch.cumprod(1 - alpha, dim=1)                   # light left after puff i
    ones = torch.ones(alpha.shape[0], 1, device=device)
    T = torch.cat([ones, left_after[:, :-1]], dim=1)               # light left before puff i

    C = (T * alpha) @ color
    return C

def render(mu, Sigma, color, opacity, order, H, W, chunk=8192):
    xy = pixel_grid(H, W)
    pieces = []
    for start in range(0, H * W, chunk):
        xy_chunk = xy[start:start + chunk]    # one slice of pixels
        piece = checkpoint(composite, xy_chunk, mu, Sigma, color, opacity, order, use_reentrant=False)
        pieces.append(piece)
    C = torch.cat(pieces, dim=0)
    return C.reshape(H, W, 3)

def get_psnr(loss):
    return -10 * torch.log10(loss)

def save_image(img, name):
    img = img.detach().clamp(0, 1).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(here, name))

# ---------- fitting the scene ----------

def init_gaussians_3d(N):
    mu3 = (torch.rand(N, 3, device=device) * 2 - 1) * 1.5      # cloud in [-1.5, 1.5]^3
    log_s = torch.log(0.08 * torch.ones(N, 3, device=device))  # small 3D blobs
    quat = torch.zeros(N, 4, device=device)
    quat[:, 0] = 1.0                                           # identity rotation
    color = torch.zeros(N, 3, device=device)                   # sigmoid -> gray
    op_raw = torch.full((N,), -2.0, device=device)             # sigmoid -> low opacity
    mu3.requires_grad_(True)
    log_s.requires_grad_(True)
    quat.requires_grad_(True)
    color.requires_grad_(True)
    op_raw.requires_grad_(True)
    return mu3, log_s, quat, color, op_raw

densify_every = 200

def fit_3d(N, iters, K, train, densify=False, max_count=None, top_frac=0.25):
    train_R, train_t, train_images = train
    H = train_images.shape[1]
    W = train_images.shape[2]
    num_views = train_images.shape[0]

    mu3, log_s, quat, color, op_raw = init_gaussians_3d(N)
    opt = torch.optim.Adam([mu3, log_s, quat, color, op_raw], lr=1e-2)
    grad_total = torch.zeros(N, device=device)        # running total of each Gaussian's push size
    grad_steps = 0

    for step in range(iters):
        i = torch.randint(0, num_views, (1,)).item()          # pick a random training camera
        Sig3 = covariance_3d(log_s.exp(), quat)
        mu2, Sig2, depth = project_gaussian(mu3, Sig3, train_R[i], train_t[i], K)
        order = torch.argsort(depth, descending=False)        # front to back
        img = render(mu2, Sig2, color.sigmoid(), op_raw.sigmoid(), order, H, W)
        loss = ((img - train_images[i]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        grad_total += mu3.grad.norm(dim=-1)
        grad_steps += 1
        opt.step()

        if (step + 1) % 100 == 0:
            print(step + 1, "N:", mu3.shape[0], "psnr on this view:", get_psnr(loss).item())

        # densify every so often, not on the last step (it would be wasted)
        if densify and (step + 1) % densify_every == 0 and (step + 1) < iters:
            avg_grad = grad_total / grad_steps
            mu3, log_s, quat, color, op_raw = densify_pass_3d(mu3, log_s, quat, color, op_raw, avg_grad, max_count, top_frac)

            # N changed, so rebuild everything that depends on it
            N = mu3.shape[0]
            opt = torch.optim.Adam([mu3, log_s, quat, color, op_raw], lr=1e-2)
            grad_total = torch.zeros(N, device=device)
            grad_steps = 0

    return mu3, log_s, quat, color, op_raw

def render_view(params, R_wc, t, K, H, W):
    mu3, log_s, quat, color, op_raw = params
    Sig3 = covariance_3d(log_s.exp(), quat)
    mu2, Sig2, depth = project_gaussian(mu3, Sig3, R_wc, t, K)
    order = torch.argsort(depth, descending=False)
    return render(mu2, Sig2, color.sigmoid(), op_raw.sigmoid(), order, H, W)

def evaluate(params, K, frames):
    # PSNR of every view in frames, one number per view
    Rs, ts, images = frames
    H = images.shape[1]
    W = images.shape[2]
    psnrs = []
    with torch.no_grad():
        for i in range(images.shape[0]):
            img = render_view(params, Rs[i], ts[i], K, H, W)
            loss = ((img - images[i]) ** 2).mean()
            psnrs.append(get_psnr(loss).item())
    return psnrs

# ---------- densification in 3D ----------

def clone_gaussians_3d(mu3, log_s, quat, color, op_raw, clone):
    # clone: (N,) True/False, True = duplicate this Gaussian in place
    mu3 = mu3.detach()
    log_s = log_s.detach()
    quat = quat.detach()
    color = color.detach()
    op_raw = op_raw.detach()

    new_mu3 = torch.cat([mu3, mu3[clone]])
    new_log_s = torch.cat([log_s, log_s[clone]])
    new_quat = torch.cat([quat, quat[clone]])
    new_color = torch.cat([color, color[clone]])
    new_op_raw = torch.cat([op_raw, op_raw[clone]])

    new_mu3.requires_grad_(True)
    new_log_s.requires_grad_(True)
    new_quat.requires_grad_(True)
    new_color.requires_grad_(True)
    new_op_raw.requires_grad_(True)
    return new_mu3, new_log_s, new_quat, new_color, new_op_raw

prune_opacity = 0.005

def prune_gaussians_3d(mu3, log_s, quat, color, op_raw):
    mu3 = mu3.detach()
    log_s = log_s.detach()
    quat = quat.detach()
    color = color.detach()
    op_raw = op_raw.detach()
    keep = op_raw.sigmoid() >= prune_opacity          # True = still visible enough

    new_mu3 = mu3[keep]
    new_log_s = log_s[keep]
    new_quat = quat[keep]
    new_color = color[keep]
    new_op_raw = op_raw[keep]

    new_mu3.requires_grad_(True)
    new_log_s.requires_grad_(True)
    new_quat.requires_grad_(True)
    new_color.requires_grad_(True)
    new_op_raw.requires_grad_(True)
    return new_mu3, new_log_s, new_quat, new_color, new_op_raw

split_scale = 1.6

def split_gaussians_3d(mu3, log_s, quat, color, op_raw, split):
    # split: (N,) True/False, True = replace this Gaussian by two smaller children
    mu3 = mu3.detach()
    log_s = log_s.detach()
    quat = quat.detach()
    color = color.detach()
    op_raw = op_raw.detach()
    keep = ~split

    parent_mu = mu3[split]                    # (K,3)
    parent_s = log_s[split].exp()             # (K,3) real sizes in world units
    parent_quat = quat[split]                 # (K,4)
    R = quaternion_to_rotation(parent_quat)   # (K,3,3) tilt of each parent

    # children land somewhere inside the parent's fuzzy volume
    noise1 = parent_s * torch.randn_like(parent_s)
    noise2 = parent_s * torch.randn_like(parent_s)
    offset1 = (R @ noise1[..., None]).squeeze(-1)     # (K,3)
    offset2 = (R @ noise2[..., None]).squeeze(-1)     # (K,3)
    child_log_s = torch.log(parent_s / split_scale)   # (K,3) smaller size

    new_mu3 = torch.cat([mu3[keep], parent_mu + offset1, parent_mu + offset2])
    new_log_s = torch.cat([log_s[keep], child_log_s, child_log_s])
    new_quat = torch.cat([quat[keep], parent_quat, parent_quat])
    new_color = torch.cat([color[keep], color[split], color[split]])
    new_op_raw = torch.cat([op_raw[keep], op_raw[split], op_raw[split]])

    new_mu3.requires_grad_(True)
    new_log_s.requires_grad_(True)
    new_quat.requires_grad_(True)
    new_color.requires_grad_(True)
    new_op_raw.requires_grad_(True)
    return new_mu3, new_log_s, new_quat, new_color, new_op_raw

size_threshold = 0.1       # world units: at or below this a struggling Gaussian is cloned, above it is split

def choose_gaussians_to_densify(avg_grad, N, max_count, top_frac):
    # avg_grad: (N,) average push size per Gaussian. returns (N,) True/False of Gaussians to densify
    threshold = torch.quantile(avg_grad, 1 - top_frac)    # only the top few percent beat this
    dense = avg_grad > threshold
    room = max(max_count - N, 0)                          # how many more Gaussians we may add
    if dense.sum().item() > room:                         # too many want in, keep the most struggling
        capped = torch.zeros_like(dense)
        if room > 0:
            best = torch.topk(avg_grad * dense, room).indices
            capped[best] = True
        dense = capped
    return dense

def densify_pass_3d(mu3, log_s, quat, color, op_raw, avg_grad, max_count, top_frac):
    N = mu3.shape[0]
    dense = choose_gaussians_to_densify(avg_grad, N, max_count, top_frac)

    max_scale = log_s.exp().max(dim=-1).values.detach()   # biggest size of each Gaussian, world units
    clone = dense & (max_scale <= size_threshold)         # small and struggling
    split = dense & (max_scale > size_threshold)          # big and struggling

    mu3, log_s, quat, color, op_raw = clone_gaussians_3d(mu3, log_s, quat, color, op_raw, clone)

    # clone added rows at the end, so the split flags need that many extra False
    extra = torch.zeros(clone.sum().item(), dtype=torch.bool, device=device)
    split = torch.cat([split, extra])

    mu3, log_s, quat, color, op_raw = split_gaussians_3d(mu3, log_s, quat, color, op_raw, split)
    mu3, log_s, quat, color, op_raw = prune_gaussians_3d(mu3, log_s, quat, color, op_raw)
    return mu3, log_s, quat, color, op_raw

def look_at(C):
    # C: (3,) camera position in the world. camera looks at the origin, world +y is up.
    forward = -C / C.norm()                               # from the camera toward the origin
    down_hint = torch.tensor([0.0, -1.0, 0.0], device=device)
    right = torch.cross(down_hint, forward, dim=0)
    right = right / right.norm()
    down = torch.cross(forward, right, dim=0)
    R_wc = torch.stack([right, down, forward], dim=0)     # three directions as the rows
    t = -R_wc @ C
    return R_wc, t

# ---------- helpers for the run blocks ----------

def save_params(params, name):
    torch.save(tuple(p.detach() for p in params), os.path.join(here, name))

def load_params(name):
    path = os.path.join(here, name)
    if not os.path.exists(path):
        print("missing", name, "- run the earlier problem first (P7 makes p7_plain_params.pt, P8 makes p8_dense_params.pt)")
        return None
    return torch.load(path, map_location=device)

def opacity_counts(params):
    # how many Gaussians are nearly see-through, and how many are mostly solid
    opacity = params[4].sigmoid()
    mean = opacity.mean().item()
    below_5 = (opacity < 0.05).sum().item()
    below_prune = (opacity < prune_opacity).sum().item()
    above_50 = (opacity > 0.5).sum().item()
    return mean, below_5, below_prune, above_50

# ---------- run blocks, one per problem ----------

def run_p7():
    # fit 4000 Gaussians to the 44 training views. took me about 5 minutes
    K, train, val, convention = load_scene()
    t0 = time.time()
    params = fit_3d(4000, 1500, K, train)
    print("seconds:", time.time() - t0)
    save_params(params, "p7_plain_params.pt")

    train_psnrs = evaluate(params, K, train)
    print("train psnr mean:", np.mean(train_psnrs), "min:", np.min(train_psnrs), "max:", np.max(train_psnrs))
    val_psnrs = evaluate(params, K, val)
    print("held-out psnr mean:", np.mean(val_psnrs))

    # render on the left, ground truth on the right
    H = train[2].shape[1]
    W = train[2].shape[2]
    with torch.no_grad():
        for i in [0, 15, 30]:
            img = render_view(params, train[0][i], train[1][i], K, H, W)
            save_image(torch.cat([img, train[2][i]], dim=1), "p7_train_view" + str(i) + ".png")

def run_p8():
    # fit with densification, 1000 -> 4000 Gaussians, and compare with the plain P7 fit (needs run_p7 first)
    K, train, val, convention = load_scene()
    plain_params = load_params("p7_plain_params.pt")
    if plain_params is None:
        return

    t0 = time.time()
    dense_params = fit_3d(1000, 1500, K, train, densify=True, max_count=4000, top_frac=0.25)
    print("seconds:", time.time() - t0)
    save_params(dense_params, "p8_dense_params.pt")

    for name, params in [("plain", plain_params), ("densified", dense_params)]:
        train_psnrs = evaluate(params, K, train)
        val_psnrs = evaluate(params, K, val)
        print(name, "N:", params[0].shape[0], "train mean:", np.mean(train_psnrs),
              "held-out mean:", np.mean(val_psnrs), "worst held-out view:", np.min(val_psnrs))
        mean, below_5, below_prune, above_50 = opacity_counts(params)
        print(name, "mean opacity:", mean, "below 0.05:", below_5, "below 0.005:", below_prune, "above 0.5:", above_50)

    # plain on the left, densified in the middle, ground truth on the right
    H = val[2].shape[1]
    W = val[2].shape[2]
    with torch.no_grad():
        for i in [0, 5, 10]:
            plain_img = render_view(plain_params, val[0][i], val[1][i], K, H, W)
            dense_img = render_view(dense_params, val[0][i], val[1][i], K, H, W)
            save_image(torch.cat([plain_img, dense_img, val[2][i]], dim=1), "p8_heldout_view" + str(i) + ".png")

def run_p9():
    # held-out numbers for both saved fits plus the orbit (needs run_p7 and run_p8 first)
    K, train, val, convention = load_scene()
    plain_params = load_params("p7_plain_params.pt")
    dense_params = load_params("p8_dense_params.pt")
    if plain_params is None or dense_params is None:
        return

    # held-out PSNR of every view, for both fits
    for name, params in [("plain", plain_params), ("densified", dense_params)]:
        val_psnrs = evaluate(params, K, val)
        print(name, "held-out psnr per view:", [round(p, 2) for p in val_psnrs])
        print(name, "held-out mean:", np.mean(val_psnrs))

    # orbit around the object: 12 frames, 4 units out, 20 degrees above the horizon
    H = val[2].shape[1]
    W = val[2].shape[2]
    radius = 4.0
    elevation = math.radians(20)
    num_frames = 12
    gif_frames = []
    with torch.no_grad():
        for k in range(num_frames):
            phi = 2 * math.pi * k / num_frames
            C = torch.tensor([radius * math.cos(elevation) * math.sin(phi),
                              radius * math.sin(elevation),
                              radius * math.cos(elevation) * math.cos(phi)], device=device)
            R_wc, t = look_at(C)
            plain_img = render_view(plain_params, R_wc, t, K, H, W)
            dense_img = render_view(dense_params, R_wc, t, K, H, W)
            save_image(plain_img, "p9_orbit_plain_" + str(k).zfill(2) + ".png")
            save_image(dense_img, "p9_orbit_dense_" + str(k).zfill(2) + ".png")

            pair = torch.cat([plain_img, dense_img], dim=1)          # plain on the left, densified on the right
            pair = (pair.clamp(0, 1) * 255).byte().cpu().numpy()
            gif_frames.append(Image.fromarray(pair))

    gif_frames[0].save(os.path.join(here, "p9_orbit.gif"), save_all=True,
                       append_images=gif_frames[1:], duration=150, loop=0)
    print("saved", num_frames, "orbit frames and p9_orbit.gif")


if __name__ == "__main__":
    # comment/uncomment whichever ones you want to run. P8 and P9 need P7's saved file, P9 also needs P8's
    # run_p7()      # about 5 minutes, saves p7_plain_params.pt
    # run_p8()      # saves p8_dense_params.pt
    run_p9()        # fast, just renders from the two saved files