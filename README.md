# Orbit recovery under the rigid motions group

MATLAB code for **orbit recovery under rigid motions**.  
The project reduces image/volume moments on **SE(n)** to moments on **SO(n)**, performs **per-ring (2‑D) / per-shell (3‑D) inversion** from {M², M³}, **synchronizes** rotations from M² only (2‑D), and **rasterizes / synthesizes** the recovered object back to the pixel/voxel grid.  
A companion suite of **statistical simulations** validates the single‑value reduction and ratio‑of‑sums estimators under noise.

---

## Repository structure

```
rigid-motion-orbit-recovery/
│
├── assets/
│   ├── albert-einstein.jpg
│   ├── monaLiza.jpg
│   ├── emdb_2984.mat
│   └── S80_ribosome.mat
│
├── full-pipeline-2D/
│   ├── bandlimit_ring_nonunit.m
│   ├── fft_index.m
│   ├── invert_all_rings_nonunit.m
│   ├── pipeline_2D_top.m
│   ├── se2_to_so2_M2M3_uniform_efficient.m
│   └── sync_ring_rotations_from_M2.m
│
├── full-pipeline-3D/
│   ├── assemble_bisys_oracle.m
│   ├── assemble_bisys_oracle_from_plan.m
│   ├── build_BW_invariants_from_translations.m
│   ├── build_bisys_sampling_plan.m
│   ├── build_for_shell_c.m
│   ├── build_gaunt_tables_w3j.m
│   ├── gaussian_shell_basis.m
│   ├── pipeline_3D_top.m
│   ├── radial_quadrature.m
│   ├── recover_so3_from_bispectrum.m
│   ├── sphharmY.m
│   ├── synth_volume_from_SH.m
│   └── volume_to_SH_coeffs_radial.m
│
└── statistical-simulation/
    ├── main_statistical_M2M3_reduction.m
    ├── run_one_trial_ratio_of_sums.m
    ├── se2_to_so2_M2M3_single.m
    └── se2_to_so2_M2M3_single_batched.m
```

> Note: `pipeline_3D_top.asv` (if present) is a MATLAB autosave and not needed.

---

## Requirements

- MATLAB **R2022a+**
- Image Processing Toolbox
- Parallel Computing Toolbox (optional; some paths use `parfor`/GPU)
- **Optional GPU**: some routines are GPU‑aware; set configuration flags accordingly.

---

## Quick start

### End‑to‑end 2‑D pipeline (SE(2) → SO(2))

Runs SE(2)→SO(2) reduction, per‑ring inversion, M²‑only synchronization, and rasterization; prints metrics and shows diagnostics/figures.

```matlab
cd full-pipeline-2D
pipeline_2D_top
```

**What it does (2‑D):**

- **Reduction:** `se2_to_so2_M2M3_uniform_efficient.m` computes spatial auto‑correlations on SE(2) and reduces them to SO(2) moments (M², M³), with an efficient implementation for dense pairwise M² and intra‑order terms.  
- **Per‑ring inversion:** `invert_all_rings_nonunit.m` reconstructs each Fourier ring using M² magnitude information and M³ (bispectrum) phase constraints under a non‑unitary model.  
- **Rotation synchronization from M²:** `sync_ring_rotations_from_M2.m` estimates per‑ring global rotations using only M² via pooled multi‑frequency constraints and a spectral initializer.  
- **Rasterization:** recovered rings are placed back onto the image grid; the pipeline reports PSNR/SSIM and visualizes intermediate diagnostics.

---

### End‑to‑end 3‑D pipeline (SE(3) → SO(3))

Validates the complete 3‑D pipeline: radial‑shell projection to spherical harmonics, **boundary‑weighted** SE(3)→SO(3) reduction to build invariant constraints, bispectrum‑based recovery of shell SH coefficients, and volume synthesis plus metrics.

```matlab
cd full-pipeline-3D
pipeline_3D_top
```

**What it does (3‑D):**

1. **Projection to shells × SH:** `volume_to_SH_coeffs_radial.m` samples the volume on spherical shells using Gaussian radial basis and evaluates spherical harmonics to produce per‑shell coefficient blocks \(F_{\ell}^m(r_c)\). It also returns the triple radial overlap tensor `Tabc` used by the bispectrum model.
2. **Gaunt tables:** `build_gaunt_tables_w3j.m` builds Gaunt matrices via closed‑form Wigner‑3j symbols.
3. **Boundary‑weighted (BW) SE(3)→SO(3) reduction:**  
   `build_BW_invariants_from_translations.m` implements the reduction by averaging over a grid of translations \(t\) with the **boundary correlation weight**
   \[ s_\delta(t) = \int_{S^2} f(t + r_0\theta)\, f(t - r_0\theta)\, d\sigma(\theta), \]
   evaluated near the spherical mask boundary \(r_0 = R(1-\delta)\). For each translated volume \(V_t\):
   - compute \(s_\delta(t)\),
   - project to shell SH \(F^{(t)}\),
   - accumulate \(s_\delta(t) \cdot G2^{(t)}\) and \(s_\delta(t)\)–weighted bispectrum rows,
   - normalize by \(\sum_t s_\delta(t)\) (with optional post‑averaging row normalization).
4. **Deterministic row plan:** `build_bisys_sampling_plan.m` creates an admissible list of \((\ell_1,\ell_2,a,b)\) tuples reused for all translations. `assemble_bisys_oracle_from_plan.m` builds an oracle Bisys with the same plan for fair row‑wise comparison.
5. **Recovery:** `recover_so3_from_bispectrum.m` solves for \(F\) using the G2 and Bisys constraints (regularized least squares, per‑\(\ell\) blocks).
6. **Synthesis and metrics:** `synth_volume_from_SH.m` renders a 3‑D volume; metrics include PSNR, SSIM, and relative \(\ell_2\) error. Optional `volshow` figures compare reference and recovered volumes.

**Main toggles in `pipeline_3D_top.m`:**
- `bw_cfg.enable` – enable/disable BW reduction.
- `use_bw_for_solver` – choose BW vs. oracle invariants for the solver.
- `bw_cfg.grid_n`, `bw_cfg.step_vox`, `bw_cfg.pad_vox` – translation grid, step, padding.
- `Lmax`, `R`, `shell_bw` – spectral/angular/radial resolution.

> Asset paths: the script expects `assets/emdb_2984.mat` and `assets/S80_ribosome.mat`. Edit the two `pth_*` variables near the top if you relocate them.

---

## Key files

### 2‑D
- `pipeline_2D_top.m` – main 2‑D experiment with figures and metrics.  
- `se2_to_so2_M2M3_uniform_efficient.m` – GPU‑aware reduction; dense pairwise M².  
- `invert_all_rings_nonunit.m` – non‑unitary per‑ring inversion from {M², M³}.  
- `sync_ring_rotations_from_M2.m` – robust M²‑only ring synchronization.

### 3‑D
- `pipeline_3D_top.m` – main 3‑D script (projection → invariants → recovery → synthesis → metrics).  
- `volume_to_SH_coeffs_radial.m` – volume → Gaussian shells × SH; returns `Tabc` and shell normalizers.  
- `synth_volume_from_SH.m` – synthesizes a 3‑D volume from recovered shell SH coefficients.  
- `recover_so3_from_bispectrum.m` – block‑wise regularized solver using G2 and Bisys.  
- `build_BW_invariants_from_translations.m` – boundary‑weighted SE(3)→SO(3) reduction (builds BW **G2** and **Bisys**).  
- `build_bisys_sampling_plan.m` – deterministic admissible row plan reused across translations.  
- `assemble_bisys_oracle.m` / `assemble_bisys_oracle_from_plan.m` – oracle Bisys (random plan / provided plan).  
- `build_gaunt_tables_w3j.m` – Gaunt matrices via closed‑form Wigner‑3j.  
- `sphharmY.m`, `gaussian_shell_basis.m`, `radial_quadrature.m` – numerical building blocks.  
- `build_for_shell_c.m` – helper to assemble rows for a single shell (sampling‑based).

### Simulations
- `main_statistical_M2M3_reduction.m` – sweeping SNR / sample budgets with plots.  
- `run_one_trial_ratio_of_sums.m`, `se2_to_so2_M2M3_single*.m` – single/batched trial drivers.

---

## Citation

If you use this code, please cite:

**Amnon Balanov, Tamir Bendory, and Dan Edidin.**  
*Orbit recovery under the rigid motions group.*

```bibtex
@article{BalanovBendoryEdidin_OrbitRecoveryRigidMotions,
  title   = {Orbit recovery under the rigid motions group},
  author  = {Balanov, Amnon and Bendory, Tamir and Edidin, Dan},
  journal = {arXiv preprint},
  year    = {2025}
}
```

---
