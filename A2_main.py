import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import time
from torch.utils.checkpoint import checkpoint
import json
import matplotlib.pyplot as plt


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()
here = os.path.dirname(os.path.abspath(__file__))     # the folder this script lives in


# ---------- P1: covariance and weight ----------

def rotation_matrix(theta):
    c = torch.cos(theta)
    s = torch.sin(theta)
    row0 = torch.stack([c, -s], dim=-1)
    row1 = torch.stack([s, c], dim=-1)
    R = torch.stack([row0, row1], dim=-2)     # (N,2,2)
    return R

def covariance_2d(scale, theta):
    # scale: (N,2) = (sx, sy), theta: (N,)
    R = rotation_matrix(theta)
    S = torch.diag_embed(scale)               # (N,2,2) sizes on the diagonal
    M = R @ S
    Sigma = M @ M.transpose(-1, -2)           # R S S^T R^T
    return Sigma

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


# ---------- P2: renderer ----------

def pixel_grid(H, W):
    ys = torch.arange(H, device=device).float()
    xs = torch.arange(W, device=device).float()
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    xy = torch.stack([grid_x, grid_y], dim=-1)    # (H,W,2)
    return xy.reshape(H * W, 2)                   # (H*W, 2)

def composite(xy, mu, Sigma, color, opacity, order):
    # xy: (P, 2) pixels to draw  ->  (P, 3) colors
    w  = gaussian_weight(xy, mu, Sigma)       # (P, N)
    alpha = opacity[None, :] * w              # (P, N)
    alpha = alpha.clamp(max=0.99)             # no puff is ever fully opaque

    alpha = alpha[:, order]                   # put the puffs in front-to-back order
    color = color[order]

    after_each = torch.cumprod(1 - alpha, dim=1)                   # leftover after puff i
    ones = torch.ones(alpha.shape[0], 1, device=device)
    T = torch.cat([ones, after_each[:, :-1]], dim=1)               # leftover before puff i

    C = (T * alpha) @ color
    return C

def render(mu, Sigma, color, opacity, order, H, W):
    xy = pixel_grid(H, W)                     # (H*W, 2)
    C = composite(xy, mu, Sigma, color, opacity, order)
    return C.reshape(H, W, 3)

def render_chunked(mu, Sigma, color, opacity, order, H, W, chunk=8192):
    xy = pixel_grid(H, W)
    pieces = []
    for start in range(0, H * W, chunk):
        xy_chunk = xy[start:start + chunk]    # one slice of pixels
        piece = checkpoint(composite, xy_chunk, mu, Sigma, color, opacity, order, use_reentrant=False)
        pieces.append(piece)
    C = torch.cat(pieces, dim=0)
    return C.reshape(H, W, 3)

test_mu = torch.zeros(3, 2, device=device)
test_Sigma = covariance_2d(torch.ones(3, 2, device=device), torch.zeros(3, device=device))
test_color = torch.eye(3, device=device)
test_opacity = torch.tensor([0.5, 0.4, 0.8], device=device)
print(render(test_mu, test_Sigma, test_color, test_opacity, torch.arange(3), 1, 1))
print("gpu memory (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)


# ---------- P3: setup helpers ----------

prune_opacity  = 0.005
split_scale    = 1.6
densify_every  = 200
size_threshold = 0.02

def load_target(image_name):
    image = Image.open(os.path.join(here, image_name)).convert("RGB")
    target = np.array(image)                              # (H, W, 3) numbers 0..255
    target = torch.from_numpy(target).float().to(device)
    target = target / 255.0                               # now 0..1, same range as render
    return target

def init_puffs(N, H, W):
    mu     = torch.rand(N, 2, device=device) * torch.tensor([W, H], device=device)
    log_s  = torch.log(0.02 * max(H, W) * torch.ones(N, 2, device=device))
    theta  = torch.zeros(N, device=device)
    color  = torch.zeros(N, 3, device=device)
    op_raw = torch.full((N,), -2.0, device=device)
    mu.requires_grad_(True)                               # these are the knobs
    log_s.requires_grad_(True)
    theta.requires_grad_(True)
    color.requires_grad_(True)
    op_raw.requires_grad_(True)
    return mu, log_s, theta, color, op_raw

def make_optimizer(mu, log_s, theta, color, op_raw):
    return torch.optim.Adam([mu, log_s, theta, color, op_raw], lr=1e-2)

def get_psnr(loss):
    return -10 * torch.log10(loss)

def save_image(img, name):
    img = img.detach().clamp(0, 1).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(here, name))


# ---------- P4: densification ----------

def split_gaussians(mu, log_s, theta, color, op_raw, split):
    # split: (N,) True/False, True = replace this puff by two smaller children
    mu = mu.detach()
    log_s = log_s.detach()
    theta = theta.detach()
    color = color.detach()
    op_raw = op_raw.detach()
    keep = ~split

    parent_mu = mu[split]                     # (K,2)
    parent_s = log_s[split].exp()             # (K,2) real sizes in pixels
    parent_theta = theta[split]               # (K,)
    R = rotation_matrix(parent_theta)         # (K,2,2) tilt of each parent

    # children land somewhere inside the parent's fuzzy area
    noise1 = parent_s * torch.randn_like(parent_s)
    noise2 = parent_s * torch.randn_like(parent_s)
    offset1 = (R @ noise1[..., None]).squeeze(-1)     # (K,2)
    offset2 = (R @ noise2[..., None]).squeeze(-1)     # (K,2)
    child_log_s = torch.log(parent_s / split_scale)   # (K,2) smaller size

    new_mu = torch.cat([mu[keep], parent_mu + offset1, parent_mu + offset2])
    new_log_s = torch.cat([log_s[keep], child_log_s, child_log_s])
    new_theta = torch.cat([theta[keep], parent_theta, parent_theta])
    new_color = torch.cat([color[keep], color[split], color[split]])
    new_op_raw = torch.cat([op_raw[keep], op_raw[split], op_raw[split]])

    new_mu.requires_grad_(True)
    new_log_s.requires_grad_(True)
    new_theta.requires_grad_(True)
    new_color.requires_grad_(True)
    new_op_raw.requires_grad_(True)
    return new_mu, new_log_s, new_theta, new_color, new_op_raw

def clone_gaussians(mu, log_s, theta, color, op_raw, clone):
    # clone: (N,) True/False, True = duplicate this puff in place
    mu = mu.detach()
    log_s = log_s.detach()
    theta = theta.detach()
    color = color.detach()
    op_raw = op_raw.detach()

    new_mu = torch.cat([mu, mu[clone]])
    new_log_s = torch.cat([log_s, log_s[clone]])
    new_theta = torch.cat([theta, theta[clone]])
    new_color = torch.cat([color, color[clone]])
    new_op_raw = torch.cat([op_raw, op_raw[clone]])

    new_mu.requires_grad_(True)
    new_log_s.requires_grad_(True)
    new_theta.requires_grad_(True)
    new_color.requires_grad_(True)
    new_op_raw.requires_grad_(True)
    return new_mu, new_log_s, new_theta, new_color, new_op_raw

def prune_gaussians(mu, log_s, theta, color, op_raw):
    mu = mu.detach()
    log_s = log_s.detach()
    theta = theta.detach()
    color = color.detach()
    op_raw = op_raw.detach()
    keep = op_raw.sigmoid() >= prune_opacity          # True = still visible enough

    new_mu = mu[keep]
    new_log_s = log_s[keep]
    new_theta = theta[keep]
    new_color = color[keep]
    new_op_raw = op_raw[keep]

    new_mu.requires_grad_(True)
    new_log_s.requires_grad_(True)
    new_theta.requires_grad_(True)
    new_color.requires_grad_(True)
    new_op_raw.requires_grad_(True)
    return new_mu, new_log_s, new_theta, new_color, new_op_raw

def pick_dense(g, N, max_count, top_frac):
    # g: (N,) average push size per puff. Returns (N,) True/False of puffs to densify
    threshold = torch.quantile(g, 1 - top_frac)       # only the top few percent beat this
    dense = g > threshold
    room = max(max_count - N, 0)                      # how many more puffs we may add
    if dense.sum().item() > room:                     # too many want in, keep the most struggling
        capped = torch.zeros_like(dense)
        if room > 0:
            best = torch.topk(g * dense, room).indices
            capped[best] = True
        dense = capped
    return dense

def densify_pass(mu, log_s, theta, color, op_raw, g, max_count, top_frac, W):
    N = mu.shape[0]
    dense = pick_dense(g, N, max_count, top_frac)

    max_scale = log_s.exp().max(dim=-1).values.detach()   # biggest size of each puff, in pixels
    clone = dense & (max_scale <= size_threshold * W)     # small and struggling
    split = dense & (max_scale > size_threshold * W)      # big and struggling

    mu, log_s, theta, color, op_raw = clone_gaussians(mu, log_s, theta, color, op_raw, clone)

    # clone added rows at the end, so the split flags need that many extra False
    extra = torch.zeros(clone.sum().item(), dtype=torch.bool, device=device)
    split = torch.cat([split, extra])

    mu, log_s, theta, color, op_raw = split_gaussians(mu, log_s, theta, color, op_raw, split)
    mu, log_s, theta, color, op_raw = prune_gaussians(mu, log_s, theta, color, op_raw)
    return mu, log_s, theta, color, op_raw


# ---------- the whole training run in one function ----------

def fit(image_name, N_start, max_count=None, steps=2000, densify=False, top_frac=0.2, chunk=None):
    target = load_target(image_name)
    H = target.shape[0]
    W = target.shape[1]

    def draw(mu, Sigma, color, opacity, order):
        if chunk is None:
            return render(mu, Sigma, color, opacity, order, H, W)
        return render_chunked(mu, Sigma, color, opacity, order, H, W, chunk)

    mu, log_s, theta, color, op_raw = init_puffs(N_start, H, W)
    depth_order = torch.arange(N_start)
    opt = make_optimizer(mu, log_s, theta, color, op_raw)
    grad_mag = torch.zeros(N_start, device=device)    # running total of each puff's push size
    grad_steps = 0


    for step in range(steps):
        Sigma = covariance_2d(log_s.exp(), theta)
        img = draw(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), depth_order)
        loss = ((img - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        grad_mag += mu.grad.norm(dim=-1)
        grad_steps += 1
        opt.step()

        at_check = (step + 1) % densify_every == 0
        if at_check:
            print(step + 1, "N:", mu.shape[0], "psnr:", get_psnr(loss).item())

        # densify every so often, not on the last step (it would be wasted)
        if densify and at_check and (step + 1) < steps:
            g = grad_mag / grad_steps
            mu, log_s, theta, color, op_raw = densify_pass(mu, log_s, theta, color, op_raw, g, max_count, top_frac, W)

            # N changed, so rebuild everything that depends on it
            N = mu.shape[0]
            depth_order = torch.arange(N)
            opt = make_optimizer(mu, log_s, theta, color, op_raw)
            grad_mag = torch.zeros(N, device=device)
            grad_steps = 0

    # final score using the finished puffs
    with torch.no_grad():
        Sigma = covariance_2d(log_s.exp(), theta)
        img = draw(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), depth_order)
        psnr = get_psnr(((img - target) ** 2).mean())
    return psnr.item(), img, mu.shape[0]


# ---------- run ----------

images = ["coffee.png", "astronaut.png", "cat.png"]
counts = [256, 1024, 4096]
results = {}            # results[(image, N)] = psnr of the plain fit
dense_results = {}      # dense_results[image] = psnr of the densified fit at 256

def image_size(name):
    image = Image.open(os.path.join(here, name))
    W, H = image.size                           # PIL gives (width, height)
    return H, W

def pick_chunk(N, H, W):
    if N * H * W > 80_000_000:                  # too big to hold all at once, draw in slices
        return 8192
    return None

def run_plain(name, N):
    H, W = image_size(name)
    t0 = time.time()
    psnr, img, n = fit(name, N_start=N, steps=2000, chunk=pick_chunk(N, H, W))
    print(">>", name, "plain N =", N, "psnr =", psnr, "seconds =", time.time() - t0)
    save_image(img, name.replace(".png", "") + "_plain_" + str(N) + ".png")
    torch.cuda.empty_cache()
    return psnr

def run_dense(name, N_final):
    H, W = image_size(name)
    t0 = time.time()
    psnr, img, n = fit(name, N_start=64, max_count=N_final, steps=2000, densify=True, chunk=pick_chunk(N_final, H, W))
    print(">>", name, "densified to", n, "psnr =", psnr, "seconds =", time.time() - t0)
    save_image(img, name.replace(".png", "") + "_dense_" + str(N_final) + ".png")
    torch.cuda.empty_cache()
    return psnr

def save_results():
    table = {}
    for (name, N), psnr in results.items():
        table["plain_" + name + "_" + str(N)] = psnr
    for name, psnr in dense_results.items():
        table["dense_" + name + "_256"] = psnr
    with open(os.path.join(here, "p5_results.json"), "w") as f:
        json.dump(table, f, indent=2)

def make_plot():
    for name in images:
        values = [results[(name, N)] for N in counts]
        plt.plot(counts, values, marker="o", label=name)
    plt.xscale("log", base=2)
    plt.xticks(counts, [str(c) for c in counts])
    plt.xlabel("number of Gaussians (N)")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.savefig(os.path.join(here, "p5_psnr_vs_N.png"), dpi=150)

for name in images:
    print("size of", name, "(H, W):", image_size(name))

# cheap jobs first: plain 256 and densified 256 for the P4 comparison
for name in images:
    results[(name, 256)] = run_plain(name, 256)
    dense_results[name] = run_dense(name, 256)
    save_results()

# the bigger counts for P5
for N in [1024, 4096]:
    for name in images:
        results[(name, N)] = run_plain(name, N)
        save_results()

print("P5: PSNR (dB) for N =", counts)
for name in images:
    print(name, [round(results[(name, N)], 2) for N in counts])
print("P4: densified vs plain at 256")
for name in images:
    print(name, round(dense_results[name], 2), "vs", round(results[(name, 256)], 2))
make_plot()
 