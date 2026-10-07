# Gaussian Splatting (A2, 15-474/674 Neural Graphics)

Fits a set of 2D Gaussians to an image, then 3D Gaussians to a multi-view scene
of spheres, with a front-to-back alpha compositing renderer and a densification
step (clone, split, prune) that grows the number of Gaussians during training.

## Setup

1. Create a virtual environment and activate it
2. `pip install torch numpy pillow matplotlib`
3. Put the files next to the scripts:
   - `coffee.png`, `astronaut.png`, `cat.png` (the three target images for the 2D part)
   - the spheres scene (provided by the course) in `spheres/spheres/`,
     with `cameras.json` and the frame images inside it

## Files

- `A2_main.py` is the 2D part, P1 through P5. It has the covariance and Gaussian weight (P1),
  the renderer (P2), the setup helpers (P3), densification (P4) and the `fit` function.
  Contains `run_p2_check()`, `run_p4()` and `run_p5()`, each runnable on its own.
- `A2_3D.py` is the 3D part, P6 through P9. It has the 3D Gaussians and projection (P6),
  the scene fitting (P7), 3D densification (P8) and the held-out evaluation with the orbit (P9).
  Contains `run_p7()`, `run_p8()` and `run_p9()`, each runnable on its own.

## How to reproduce results

    python A2_main.py
    python A2_3D.py

Both scripts have an `if __name__ == "__main__":` block at the bottom. Comment or uncomment
the run functions you want.

- `A2_main.py` runs `run_p2_check()` and `run_p4()` by default (a few minutes).
  `run_p5()` is commented out because the whole N = 256, 1024, 4096 grid took about
  1.5 hours on my GPU. Uncomment it to redo the P5 table and plot. If `run_p4()` runs
  in the same call, the N = 256 plain fits are reused instead of retrained.
- `A2_3D.py` runs only `run_p9()` by default, which renders from two saved files.
  To start from scratch, run `run_p7()` first (about 5 minutes, saves `p7_plain_params.pt`),
  then `run_p8()` (saves `p8_dense_params.pt`), then `run_p9()`.

## Outputs

- P4 and P5: `<image>_plain_<N>.png`, `<image>_dense_256.png`, `p5_results.json`, `p5_psnr_vs_N.png`
- P7: `p7_train_view0.png`, `p7_train_view15.png`, `p7_train_view30.png` (render on the left, ground truth on the right)
- P8: `p8_heldout_view0.png`, `p8_heldout_view5.png`, `p8_heldout_view10.png`
  (plain, densified, ground truth, left to right)
- P9: `p9_orbit_plain_XX.png`, `p9_orbit_dense_XX.png` for the 12 orbit frames, and `p9_orbit.gif`
  (plain on the left, densified on the right)

Every run prints PSNR to the console.

## Note

I did not set a random seed, so the results change a little from run to run. Two runs
of the same 2D fit differed by about 0.3 dB. The numbers in my writeup each come from
a single run.

 