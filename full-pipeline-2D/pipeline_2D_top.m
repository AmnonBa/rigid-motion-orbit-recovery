function pipeline_2D_top()
%PIPELINE_2D_TOP  End-to-end SE(2)→SO(2) pipeline with reduction, inversion, sync, and rasterization.
%   PIPELINE_2D_TOP() runs a complete 2-D reconstruction experiment starting from a single
%   image on a padded disk. It:
%       1) Configures geometry and reduction knobs.
%       2) (Optional) Applies an initial angular bandlimit on all rings.
%       3) Reduces SE(2) statistics to SO(2): computes intra-ring M^{(3)}, M^{(2)}, and
%          ALL pairwise M^{(2)} (unitary φ-domain).
%       4) Converts unitary → non-unitary normalizations when needed.
%       5) Chooses per-ring bandlimits K_r from M^{(2)}.
%       6) Builds bispectrum blocks and inverts each ring (non-unitary inverter).
%       7) Synchronizes ring rotations using M^{(2)} only (multi-k pooling + spectral init).
%       8) Rasterizes reconstructed rings back to the image plane and evaluates quality.
%       9) (Optionally) Repeats inversion using “DIRECT” (truth) moments for comparison.
%      10) Prints diagnostics and shows summary plots/metrics.
%
%   The script is designed for **reproducible experiments** comparing a reduction-based
%   reconstruction path (“RED”) against a direct moment path (“DIR”), while keeping all
%   synchronization based on the *reduced* pairwise M^{(2)} (to avoid leakage of truth).
%
%   INPUTS / OUTPUTS
%     • This is a top-level driver with no input arguments and no return value. It prints
%       metrics to the console and opens figures. Intermediate arrays are kept local.
%
%   NORMALIZATION CONVENTIONS
%     • “UNITARY” refers to FFT/IFFT with 1/√N factors (used inside reducers & diagnostics).
%     • “NON-UNITARY” refers to MATLAB’s default FFT/IFFT (used by ring inversion).
%     • The pipeline converts between these where noted to keep formulas simple and stable.
%
%   DEPENDENCIES (required on path)
%     Core:
%       - se2_to_so2_M2M3_uniform_efficient      (SE(2)→SO(2) reducer; GPU-aware)
%       - invert_all_rings_nonunit                (per-ring non-unitary inverter)
%       - sync_ring_rotations_from_M2             (M²-only inter-ring synchronization)
%     Utilities:
%       - bandlimit_ring_nonunit, fft_index
%       - sample_rings_RxNphi, rings_to_image_RxNphi
%       - compute_M23_by_definition_all, compute_pairwise_M2_unit_from_rings
%       - disk_mask, fro_rel, align_by_rotation, etc. (all defined below)
%
%   TYPICAL USE
%     Simply run:
%         >> pipeline_2D_top
%     Optionally edit the image path and the cfg / inv structs near the top of this file.
%
%   NOTES
%     • For fair comparisons, inter-ring synchronization (rotation estimation) *always*
%       uses the REDUCED pairwise M^{(2)} tensor, even when inverting from DIRECT moments.
%     • The initial angular bandlimit (cfg.init_bandlimit_phiK via inv.Kmin) can stabilize
%       reduction when the input image has strong high-frequency content.
%     • Figures show both absolute reconstructions and per-ring/pair diagnostics.
%
%   Author: <your name>, <affiliation> (<year>)
%
%   -------------------------------------------------------------------------
%   End-to-end: SE(2) → SO(2) {M^{(3)}, M^{(2)}, M^{(2)}_{pairs}} reduction
%   → per-ring inversion → M^2-only ring synchronization (multi-k) → rasterize.
%   -------------------------------------------------------------------------

%% -------------------- Geometry & reduction knobs --------------------
cfg = struct();
cfg.R      = 200;            % disk radius (pixels) inside padded canvas
cfg.pad    = 10;             % padding outside disk
cfg.Nphi   = 360;            % angular samples per ring (φ)

% Reduction (uniform rotation law; no α loop)
cfg.Nphi_os  = cfg.Nphi;
cfg.Ntheta   = [];           % auto
cfg.margin_ring = 0.2;
cfg.max_r    = cfg.R - 1;    % typically skip boundary ring
cfg.allow_partial_last_ring = false;
cfg.delta_pix = 0.005;

cfg.limit_translations = true;
cfg.trans_radius = 4;
cfg.prune_by_sabs = false;
cfg.sabs_quantile = 0.6;
cfg.max_translations = [];

% Impl/perf
cfg.useGPU   = false;
cfg.verbose  = true;
cfg.phi_chunk = 120;
cfg.r_chunk   = 12;
cfg.ringMethod = 'cubic';
cfg.bndMethod  = 'cubic';

% NEW: ask reducer to compute all inter-ring M^2 pairs
cfg.compute_M2_pairs = true;

% Exports: keep UNITARY normalization in M2_dense (set false)
cfg.match_direct_norm = false;

% Ask reducer for M^3 triplets equal to (r,r,r) for all r (to validate intra M^3)
cfg.triplets_mode = 'explicit';
% triplets are 0-based inside the reducer; will set below after we know max_r

%% -------------------- Inversion knobs (non-unitary inverter) --------
inv = struct();
inv.Nphi            = cfg.Nphi;
inv.energy_keep     = 0.995;   % per-ring: keep this fraction of |X_m|^2
inv.Kmin            = 128*2;   % minimum |m|
inv.Kmax_frac       = 0.99;    % cap at fraction of Nyquist
inv.print_per_ring  = true;
cfg.init_bandlimit_phiK = inv.Kmin;

%% -------------------- Load image on padded disk ---------------------
% (Edit path as needed; function below embeds/crops to a disk on the canvas)
% --- Load image from a relative path (one folder up: ../assets/) ---
thisDir = fileparts(mfilename('fullpath'));
assetsDir = fullfile(thisDir, '..', 'assets');
% Choose the image:
% imgName = 'monaLiza.jpg';
imgName = 'albert-einstein.jpg';
img_path = fullfile(assetsDir, imgName);
assert(exist(img_path, 'file') == 2, 'Image not found: %s', img_path);

[f, inside] = load_embed_disk_image_local(img_path, cfg.R, cfg.pad);
S  = size(f,1); ctr = (S+1)/2; mask = disk_mask(S, cfg.R);
fprintf('Canvas size %dx%d, disk radius R=%d (pad=%d)\n', S, S, cfg.R, cfg.pad);

% --- NEW: apply initial angular bandlimit on rings of the loaded image ---
if true && ~isempty(cfg.init_bandlimit_phiK) && cfg.init_bandlimit_phiK >= 0
    K0   = min(cfg.init_bandlimit_phiK, floor(cfg.Nphi/2)-1); % Nyquist guard
    nr0  = cfg.max_r - 2;
    Rings0 = sample_rings_RxNphi(f, cfg.Nphi, cfg.max_r, ctr);      % [nr0 x Nphi]
    RingsBL = zeros(size(Rings0));
    for ir = 1:nr0
        RingsBL(ir,:) = bandlimit_ring_nonunit(Rings0(ir,:), K0, cfg.Nphi);
    end
    f = rings_to_image_RxNphi(RingsBL, S, ctr, cfg.max_r);   % back to image
    % Re-normalize on the disk (keep zeros outside)
    f(~inside) = 0;
    f(inside)  = mat2gray(f(inside));
    fprintf('[init-bandlimit] Applied angular bandlimit |m|<=%d (Nphi=%d)\n', K0, cfg.Nphi);
end

%% -------------------- Reduction: SE(2) → {M3, M2, M2_pairs} (UNITARY)
tic;
% NEW API: returns intra-ring M3, intra-ring M2, and pairwise M2 (all unitary)
max_r = cfg.max_r;
triplets_rrr = [(0:max_r).', (0:max_r).', (0:max_r).'];   % 0-based (r,r,r)
cfg.triplets_list = triplets_rrr;
[M3_red, ~, meta, M2_dense, ~] = ...
    se2_to_so2_M2M3_uniform_efficient(f, cfg, [], [], []);
nr = meta.max_r + 1;
fprintf('[reduction] SE2→SO2 M^{(3)}+M^{(2)}+M^{(2)}_{pairs}: %.3fs | rings 0..%d\n', toc, meta.max_r);

%% -------------------- Build reducer's intra M2 & M3 (UNITARY) ----------
% Intra M^2 from the diagonal of the dense pairs cube:
M2_red_unit = zeros(cfg.Nphi, nr);
for r = 1:nr
    M2_red_unit(:,r) = squeeze(M2_dense(r,r,:));       % unitary by cfg.match_direct_norm=false
end

% Intra M^3: M3_red contains only (r,r,r) triplets; remap to a [Nphi x Nphi x nr] array
M3_red_unit = zeros(cfg.Nphi, cfg.Nphi, nr);
T = M3_red.triplets;     % 0-based, sorted; here each row should be [r r r]
assert(size(T,1) == nr, 'Expected one triplet per ring for (r,r,r) explicit mode.');
for t = 1:size(T,1)
    r0 = T(t,1);                     % 0-based
    assert(all(T(t,:)==r0), 'Triplet is not (r,r,r).');
    M3_red_unit(:,:,r0+1) = M3_red.data(:,:,t);
end

%% -------------------- “Truth” (UNITARY) for diagnostics ------------
[M3_true_unit, M2_true_unit] = compute_M23_by_definition_all(f, cfg.R, cfg.pad, cfg.Nphi, meta.max_r);

% Inter-ring truth (for all ring pairs) for validation only
Rings_ref = sample_rings_RxNphi(f, cfg.Nphi, meta.max_r, ctr); % [nr × Nphi]
M2pairs_true_unit = compute_pairwise_M2_unit_from_rings(Rings_ref, cfg.Nphi); % [nr x nr x Nphi]

%% -------------------- Diagnostics on UNITARY ------------------------
print_ring_diagnostics_23(M3_red_unit, M3_true_unit, 'M^{(3)} (unitary)');
print_ring_diagnostics_2 (M2_red_unit, M2_true_unit, 'M^{(2)} (unitary)');

% NEW: validate ALL inter-ring M^2 pairs
pair_diag = print_pairwise_M2_diagnostics_unit(M2_dense, M2pairs_true_unit, 'M^{(2)} pairs (unitary)');

%% -------------------- Convert UNITARY → NON-UNITARY -----------------
Nphi = cfg.Nphi; N = Nphi;
M2_red  = M2_red_unit  * sqrt(N);
M3_red  = M3_red_unit  * sqrt(N);
M2_true = M2_true_unit * sqrt(N);
M3_true = M3_true_unit * sqrt(N);

%% -------------------- Adaptive K_r via |X|^2 from M2_non ------------
K_r = choose_K_per_ring_from_M2_nonunit(M2_red, inv);       % uses fft(M2_non)
fprintf('[K_r] min=%d  median=%d  max=%d\n', min(K_r), median(K_r), max(K_r));

%% -------------------- Build bispectrum blocks B from M3_non ---------
B_all_red  = build_B_blocks_from_M3_non(M3_red,  Nphi, K_r);
B_all_true = build_B_blocks_from_M3_non(M3_true, Nphi, K_r); % for DIRECT branch

%% -------------------- Invert per ring (REDUCTION path) --------------
fprintf('\n--- Inversion from reduction {M^{(3)}, M^{(2)}} ---\n');
[ringsA_hat, metricsA] = invert_all_rings_nonunit(M2_red, B_all_red, Rings_ref, K_r, inv); % diagnostics only

%% -------------------- Inter-ring synchronization from reduced M^2 ---
% Build M^2 cell directly from the REDUCED pair tensor (unitary φ-domain)
M2_cell = cell(nr,nr);
for i = 1:nr
    for j = 1:nr
        M2_cell{i,j} = squeeze(M2_dense(i,j,:)).';  % 1×Nphi
    end
end

% Robust M²-only synchronization (multi-k pooling + spectral init)
Kmax_sync = min(K_r);
opts_sync = struct('Kmax', 1, 'min_edge_weight', 1e-12, 'verbose', true);
[thetaA,~] = sync_ring_rotations_from_M2(ringsA_hat, M2_cell, Nphi, opts_sync);

% Apply rotations to all rings
ringsA_sync = zeros(size(ringsA_hat));
for ir = 1:nr
    ringsA_sync(ir,:) = apply_fractional_rotation_ring_nonunit(ringsA_hat(ir,:), thetaA(ir), Nphi);
end

% Rasterize and LS-scale on disk (only for evaluation)
fA = rings_to_image_RxNphi(ringsA_sync, S, ctr, meta.max_r);
[fA, best_deg_A] = align_by_rotation(f, fA, cfg.R); 
numA = sum( (f .* fA) .* mask, 'all' ); denA = sum( (fA.^2) .* mask, 'all' ) + 1e-12;
fA  = (numA/denA) * fA;
relA = fro_rel(f, fA, cfg.R);

% Back-check: compute UNITARY moments from fA, compare to REDUCTION (NON-UNITARY)
[M3_from_fA_unit, M2_from_fA_unit] = compute_M23_by_definition_all(fA, cfg.R, cfg.pad, cfg.Nphi, meta.max_r);
M3_from_fA = M3_from_fA_unit * sqrt(N);  M2_from_fA = M2_from_fA_unit * sqrt(N);
report_backcheck_23(M3_from_fA, M3_red, 'M^{(3)} (non-unitary) vs reduction');
report_backcheck_2 (M2_from_fA, M2_red, 'M^{(2)} (non-unitary) vs reduction');
fprintf('  [RED] image rot(best)=%.1f°, rel err=%.3f%%\n', best_deg_A, 100*relA);

%% -------------------- Optional: invert DIRECT {M2,M3} ----------------
fprintf('\n--- Inversion from DIRECT {M^{(3)}, M^{(2)}} ---\n');
[ringsB_hat, metricsB] = invert_all_rings_nonunit(M2_true, B_all_true, Rings_ref, K_r, inv); % diagnostics only

% IMPORTANT: reuse the **same reduced** M2_cell for synchronization (no truth!)
thetaB = sync_ring_rotations_from_M2(ringsB_hat, M2_cell, Nphi, opts_sync);

ringsB_sync = zeros(size(ringsB_hat));
for ir = 1:nr
    ringsB_sync(ir,:) = apply_fractional_rotation_ring_nonunit(ringsB_hat(ir,:), thetaB(ir), Nphi);
end

fB = rings_to_image_RxNphi(ringsB_sync, S, ctr, meta.max_r);
[fB, best_deg_B] = align_by_rotation(f, fB, cfg.R); 
numB = sum( (f .* fB) .* mask, 'all' ); denB = sum( (fB.^2) .* mask, 'all' ) + 1e-12;
fB  = (numB/denB) * fB;
relB = fro_rel(f, fB, cfg.R);

[M3_from_fB_unit, M2_from_fB_unit] = compute_M23_by_definition_all(fB, cfg.R, cfg.pad, cfg.Nphi, meta.max_r);
M3_from_fB = M3_from_fB_unit * sqrt(N);  M2_from_fB = M2_from_fB_unit * sqrt(N);
report_backcheck_23(M3_from_fB, M3_true, 'M^{(3)} (non-unitary) vs direct');
report_backcheck_2 (M2_from_fB, M2_true, 'M^{(2)} (non-unitary) vs direct');
fprintf('  [DIR] image rot(best)=%.1f°, rel err=%.3f%%\n', best_deg_B, 100*relB);

%% -------------------- Visuals + final reconstruction metrics ---------
% Metrics on the masked disk only
M = mask; npx = sum(M(:));
vec = @(X) X(M>0);

% A (reduction)
errA  = (fA - f) .* M;
mseA  = sum(errA(:).^2) / max(npx,1);
rmseA = sqrt(mseA);
maeA  = sum(abs(errA(:))) / max(npx,1);
relF_A = relA;                                   % already computed
corrA = corr(vec(f), vec(fA));                   % Pearson corr on disk
if exist('psnr','file')==2
    psnrA = psnr(fA.*M, f.*M);                   % OK: same mask zeros outside
else
    psnrA = 10*log10(1 / max(mseA,1e-20));       % images are in [0,1]
end
if exist('ssim','file')==2
    ssimA = ssim(fA, f);                         % both are [0,1], zeros outside disk
else
    ssimA = NaN;
end

% B (direct)
errB  = (fB - f) .* M;
mseB  = sum(errB(:).^2) / max(npx,1);
rmseB = sqrt(mseB);
maeB  = sum(abs(errB(:))) / max(npx,1);
relF_B = relB;                                   % already computed
corrB = corr(vec(f), vec(fB));
if exist('psnr','file')==2
    psnrB = psnr(fB.*M, f.*M);
else
    psnrB = 10*log10(1 / max(mseB,1e-20));
end
if exist('ssim','file')==2
    ssimB = ssim(fB, f);
else
    ssimB = NaN;
end

% Console summary
fprintf('\n[FINAL METRICS — Reduction]\n');
fprintf('  PSNR=%.2f dB | SSIM=%s | relF=%.3f%% | corr=%.4f | MAE=%.3e | RMSE=%.3e\n', ...
    psnrA, num2str(ssimA,'%.4f'), 100*relF_A, corrA, maeA, rmseA);
fprintf('[FINAL METRICS — Direct]\n');
fprintf('  PSNR=%.2f dB | SSIM=%s | relF=%.3f%% | corr=%.4f | MAE=%.3e | RMSE=%.3e\n', ...
    psnrB, num2str(ssimB,'%.4f'), 100*relF_B, corrB, maeB, rmseB);

% Side-by-side image display (original, recon A/B, and errors)
figure('Name','Original vs Reconstructions (Reduction / DIRECT)');
colormap gray;
subplot(2,3,1); imagesc(f);  axis image off; colorbar; title('Original');
subplot(2,3,2); imagesc(fA); axis image off; colorbar;
title(sprintf('Recon A (RED) | rot=%.1f^\\circ', best_deg_A));
subplot(2,3,3); imagesc((fA-f).*M); axis image off; colorbar;
title(sprintf('Error A | PSNR=%.2f dB | SSIM=%s', psnrA, num2str(ssimA,'%.3f')));

subplot(2,3,4); imagesc(f);  axis image off; colorbar; title('Original');
subplot(2,3,5); imagesc(fB); axis image off; colorbar;
title(sprintf('Recon B (DIR) | rot=%.1f^\\circ', best_deg_B));
subplot(2,3,6); imagesc((fB-f).*M); axis image off; colorbar;
title(sprintf('Error B | PSNR=%.2f dB | SSIM=%s', psnrB, num2str(ssimB,'%.3f')));

%% -------------------- Plots -----------------------------------------
r_show = round(meta.max_r/2);
figure('Name','SO(2) M^{(3)} (unitary): reduction vs truth (middle ring)');
subplot(1,3,1); imagesc(M3_true_unit(:,:,r_show)); axis image off; colorbar; title('M^{(3)} truth (unitary)');
subplot(1,3,2); imagesc(M3_red_unit(:,:,r_show));  axis image off; colorbar; title('M^{(3)} reduced (unitary)');
subplot(1,3,3); imagesc(M3_red_unit(:,:,r_show)-M3_true_unit(:,:,r_show)); axis image off; colorbar; title('Diff');

figure('Name','SO(2) M^{(2)} (unitary): reduction vs truth (middle ring)');
subplot(1,3,1); plot(M2_true_unit(:,r_show),'-'); grid on; title('M^{(2)} truth (unitary)'); xlabel('\Delta\phi idx');
subplot(1,3,2); plot(M2_red_unit(:,r_show),'-');  grid on; title('M^{(2)} reduced (unitary)'); xlabel('\Delta\phi idx');
subplot(1,3,3); plot(M2_red_unit(:,r_show)-M2_true_unit(:,r_show),'-'); grid on; title('Diff (unitary)'); xlabel('\Delta\phi idx');

% Random pair (i,j) view for pairwise M2
rng(0);
i_show = randi([1 nr]); j_show = randi([1 nr]);
figure('Name','SO(2) M^{(2)} pairs (unitary): reduction vs truth (random pair)');
subplot(1,3,1); plot(squeeze(M2pairs_true_unit(i_show,j_show,:)),'-'); grid on; ...
    title(sprintf('M^{(2)} truth (r_i=%d,r_j=%d)', i_show-1, j_show-1)); xlabel('\Delta\phi idx');
subplot(1,3,2); plot(squeeze(M2_dense(i_show,j_show,:)),'-'); grid on; title('M^{(2)} reduced'); xlabel('\Delta\phi idx');
subplot(1,3,3); plot(squeeze(M2_dense(i_show,j_show,:) - M2pairs_true_unit(i_show,j_show,:)),'-'); ...
    grid on; title('Diff'); xlabel('\Delta\phi idx');

% Diagonal (i=i) equals intra-ring M2
i_diag = r_show;
figure('Name','Pairwise vs Intra-ring M^{(2)} consistency (unitary)');
plot(squeeze(M2pairs_true_unit(i_diag,i_diag,:)),'k-','DisplayName','pair truth (i,i)'); hold on;
plot(M2_true_unit(:,i_diag),'k--','DisplayName','intra truth (i)'); 
plot(squeeze(M2_dense(i_diag,i_diag,:)),'r-','DisplayName','pair reduced (i,i)');
plot(M2_red_unit(:,i_diag),'r--','DisplayName','intra reduced (i)');
grid on; legend; xlabel('\Delta\phi idx');
title(sprintf('Consistency check (r=%d)', i_diag-1));

%% -------------------- Summary prints --------------------------------
fprintf('\n[INTRA] M3: mean/median/max rel-err = %.3e / %.3e / %.3e\n', ...
    mean_rel_err(M3_red_unit,M3_true_unit), median_rel_err(M3_red_unit,M3_true_unit), max_rel_err(M3_red_unit,M3_true_unit));
fprintf('[INTRA] M2: mean/median/max rel-err = %.3e / %.3e / %.3e\n', ...
    mean_rel_err(M2_red_unit,M2_true_unit), median_rel_err(M2_red_unit,M2_true_unit), max_rel_err(M2_red_unit,M2_true_unit));
fprintf('[PAIRS] M2: mean/median/max rel-err = %.3e / %.3e / %.3e\n', ...
    pair_diag.pairs.mean_rel, pair_diag.pairs.median_rel, pair_diag.pairs.max_rel);

% Ring-wise summary (diagnostics)
fprintf('\n[RING METRICS — reduction]\n');
print_ring_metrics_summary(metricsA);
fprintf('[RING METRICS — direct]\n');
print_ring_metrics_summary(metricsB);
end

function x_shift = apply_fractional_rotation_ring_nonunit(x_row, theta, Nphi)
% Fractional circular shift corresponding to rotation by 'theta' radians.
% This is correct: a shift by s samples multiplies by exp(-i 2π k s/N) = exp(-i k theta).
X = fft(x_row, Nphi);
k = 0:Nphi-1;
phase = exp(-1i * theta * k);
X = X .* phase;
x_shift = real(ifft(X, 'symmetric'));
if ~isreal(x_shift), x_shift = real(x_shift); end
end

%% ======================= Builders / selection ========================
function K_r = choose_K_per_ring_from_M2_nonunit(M2_stack_non, inv)
Nphi = inv.Nphi; Kmax_abs = floor((Nphi/2)*inv.Kmax_frac);
nr   = size(M2_stack_non,2);
K_r  = zeros(nr,1);
for ir=1:nr
    M2_phi = M2_stack_non(:,ir).';
    M2_hat = real(fft(M2_phi, Nphi));   % = |X|^2
    E0 = M2_hat(1);
    Em = zeros(1, Kmax_abs);
    for m=1:Kmax_abs
        Em(m) = M2_hat(fft_index( m,Nphi)) + M2_hat(fft_index(-m,Nphi));
    end
    Ecum = E0 + cumsum(Em);
    Etot = E0 + sum(Em);
    targ = inv.energy_keep * max(Etot, eps);
    kstar = find(Ecum >= targ, 1, 'first'); if isempty(kstar), kstar = Kmax_abs; end
    K_r(ir) = max(inv.Kmin, min(kstar, Kmax_abs));
end
end

function B_cells = build_B_blocks_from_M3_non(M3_stack_non, Nphi, K_r)
[n1, n2, nr] = size(M3_stack_non);
if n1 ~= Nphi || n2 ~= Nphi
    error('Size mismatch: expected Nphi×Nphi×nr, got %dx%dx%d', n1, n2, nr);
end
K_r   = K_r(:);
Kuniq = unique(K_r.');  maxK  = max(Kuniq);
linIdxCache = cell(maxK+1, 1);
for K = Kuniq
    modes = -K:K;
    [M1, M2] = ndgrid(modes, modes);
    rows = mod(M1, Nphi) + 1;
    cols = mod(M2, Nphi) + 1;
    linIdxCache{K+1} = sub2ind([Nphi, Nphi], rows, cols);
end
B_cells = cell(nr,1);
for ir = 1:nr
    K  = K_r(ir);
    li = linIdxCache{K+1};
    Bfull = fft2(M3_stack_non(:,:,ir));    % non-unitary
    B     = zeros(2*K+1, 2*K+1);
    B(:)  = Bfull(li);
    B_cells{ir} = B;
end
end

%% ======================= Sampling / rasterizing ======================
function Rings = sample_rings_RxNphi(f, Nphi, max_r, ctr)
phi = linspace(0, 2*pi*(1-1/Nphi), Nphi);
nr  = max_r + 1;
Rings = zeros(nr, Nphi);
for ir=1:nr
    r = ir-1;
    x = ctr + r*cos(phi);  y = ctr + r*sin(phi);
    Rings(ir,:) = interp2(f, x, y, 'cubic', 0);
end
end

function img = rings_to_image_RxNphi(ringsRXN, S, ctr, max_r)
[nr, Nphi] = size(ringsRXN);
[xx,yy] = meshgrid(1:S,1:S);
dx = xx - ctr; dy = yy - ctr;
r  = hypot(dx,dy);
th = atan2(dy,dx); th(th<0) = th(th<0) + 2*pi;
t_idx = round(th/(2*pi)*(Nphi-1)) + 1;
t_idx = min(max(t_idx,1), Nphi);
r0 = floor(r); r1 = r0 + 1; alpha = r - r0;
r0 = min(max(r0,0), max_r); r1 = min(max(r1,0), max_r);
sample = @(ri,ti) sample_ring_RXN(ringsRXN, ri, ti);
img = (1-alpha).*sample(r0,t_idx) + alpha .* sample(r1,t_idx);
img(r>max_r) = 0;
end

function vals = sample_ring_RXN(ringsRXN, r_ind, t_ind)
[nr, Nphi] = size(ringsRXN);
vals = zeros(size(r_ind));
mask = (r_ind>=1);
if any(mask(:))
    lin = sub2ind([nr,Nphi], r_ind(mask), t_ind(mask));
    tmp = ringsRXN(lin);
    vals(mask) = tmp;
end
end

%% ======================= Pairwise M2 (unitary) from rings ===========
function M2_pair_unit = compute_pairwise_M2_unit_from_rings(RingsRXN, Nphi)
% RingsRXN: [nr × Nphi], real-valued.
% Output: M2_pair_unit [nr × nr × Nphi], unitary scaling in φ-domain.
nr = size(RingsRXN,1);
M2_pair_unit = zeros(nr,nr,Nphi);
for i=1:nr
    xi = RingsRXN(i,:);
    Xi = fft(xi)/sqrt(Nphi);  % unitary spectrum
    for j=1:nr
        xj = RingsRXN(j,:);
        Xj = fft(xj)/sqrt(Nphi);
        Cij = Xi .* conj(Xj);                       % cross-power (phase = ϕ_i - ϕ_j)
        M2_pair_unit(i,j,:) = real(ifft(Cij))*sqrt(Nphi);  % unitary φ-domain
    end
end
end


%% ======================= Diagnostics / utils =========================
function print_ring_diagnostics_23(Ared, Atru, name)
nr=size(Ared,3);
rel=zeros(nr,1); als=zeros(nr,1); cr=zeros(nr,1);
fprintf('%s ring diagnostics (r, alpha_lsq, corr, rel_err):\n', name);
for ir=1:nr
    A = Ared(:,:,ir); B = Atru(:,:,ir);
    rel(ir) = norm(A(:)-B(:))/max(norm(B(:)),1e-30);
    a=B(:); b=A(:); als(ir)=(a'*b)/max(a'*a,1e-30);
    a0=a-mean(a); b0=b-mean(b); cr(ir)=(a0'*b0)/max(norm(a0)*norm(b0),1e-30);
    fprintf('  r=%3d | α=%.4f  corr=%.4f  rel=%.3e\n', ir-1, als(ir), cr(ir), rel(ir));
end
fprintf('%s rel err: mean=%.3e  median=%.3e  max=%.3e\n', name, mean(rel), median(rel), max(rel));
end

function print_ring_diagnostics_2(Ared, Atru, name)
nr=size(Ared,2);
rel=zeros(nr,1); als=zeros(nr,1); cr=zeros(nr,1);
fprintf('\n%s ring diagnostics (r, alpha_lsq, corr, rel_err):\n', name);
for ir=1:nr
    A = Ared(:,ir); B = Atru(:,ir);
    rel(ir) = norm(A(:)-B(:))/max(norm(B(:)),1e-30);
    a=B(:); b=A(:); als(ir)=(a'*b)/max(a'*a,1e-30);
    a0=a-mean(a); b0=b-mean(b); cr(ir)=(a0'*b0)/max(norm(a0)*norm(b0),1e-30);
    fprintf('  r=%3d | α=%.4f  corr=%.4f  rel=%.3e\n', ir-1, als(ir), cr(ir), rel(ir));
end
fprintf('%s rel err: mean=%.3e  median=%.3e  max=%.3e\n', name, mean(rel), median(rel), max(rel));
end

function report_backcheck_23(A, B, tag)
nr=size(B,3); rel=zeros(nr,1);
for ir=1:nr
    X=A(:,:,ir); Y=B(:,:,ir);
    rel(ir)=norm(X(:)-Y(:))/max(norm(Y(:)),1e-30);
end
fprintf('Back-check %s: mean=%.3e, median=%.3e, max=%.3e\n', tag, mean(rel), median(rel), max(rel));
end

function report_backcheck_2(A, B, tag)
nr=size(B,2); rel=zeros(nr,1);
for ir=1:nr
    X=A(:,ir); Y=B(:,ir);
    rel(ir)=norm(X(:)-Y(:))/max(norm(Y(:)),1e-30);
end
fprintf('Back-check %s: mean=%.3e, median=%.3e, max=%.3e\n', tag, mean(rel), median(rel), max(rel));
end

function print_ring_metrics_summary(m)
fprintf('relL2 (med)=%.3e | corr (med)=%.4f | SNR (med)=%.1f dB | eq_rot (med)=%.1f°\n', ...
    median(m(:,1)), median(m(:,2)), median(m(:,3)), median(m(:,4)));
end


function [f_rot_best, best_deg] = align_by_rotation(f_ref, f_hat, R)
[H,~] = size(f_ref); mask = disk_mask(H, R);
angles = 0:1:359; 
best_val = -inf; 
best_deg = 0; 
for a = angles
    g = imrotate(f_hat, a, 'bilinear', 'crop');
    val = sum(g(:).*mask(:).*f_ref(:).*mask(:));
    if val > best_val
        best_val = val; 
        best_deg = a; 
    end
end
f_rot_best = imrotate(f_hat, best_deg, 'bilinear', 'crop');
end

function m = disk_mask(S, R)
ctr=(S+1)/2; [X,Y]=meshgrid(1:S,1:S); m = hypot(X-ctr,Y-ctr) <= R;
end

function e = fro_rel(f, g, R)
M = disk_mask(size(f,1), R);
e = norm((g-f).*M,'fro')/(norm(f.*M,'fro')+1e-12);
end

%% -------- Local “truth” calculators (UNITARY) ----------
function [M3, M2] = compute_M23_by_definition_all(f, R, pad, Nphi, max_r)
if nargin < 5 || isempty(max_r), max_r = R; end
S = 2*(R+pad)+1; ctr=(S+1)/2; nr = max_r+1;
phi  = linspace(0, 2*pi*(1-1/Nphi), Nphi);

% sample rings
Rings = zeros(Nphi, nr);
for ir=1:nr
    r=ir-1; x=ctr+r*cos(phi); y=ctr+r*sin(phi);
    Rings(:,ir)=interp2(f,x,y,'cubic',0);
end

M3 = zeros(Nphi,Nphi,nr);
M2 = zeros(Nphi,     nr);

[m1g,m2g]=ndgrid(0:Nphi-1,0:Nphi-1);
IDX2D = mod(-(m1g+m2g),Nphi)+1;
IDX1D = mod(-(0:Nphi-1),Nphi)+1;

for ir=1:nr
    c = fft(Rings(:,ir))/sqrt(Nphi);     % UNITARY ring spectrum
    B = (c*c.').*c(IDX2D);
    M3(:,:,ir) = real(ifft2(B))*Nphi;    % 2D unitary inverse
    Sspec = c .* c(IDX1D);
    M2(:,ir)   = real(ifft(Sspec))*sqrt(Nphi);
end
end

function [f, inside] = load_embed_disk_image_local(img_path, R, pad)
S = 2*(R + pad) + 1; ctr = (S+1)/2;
[X,Y] = meshgrid(1:S,1:S);
inside = hypot(X-ctr, Y-ctr) <= R;
if exist(img_path,'file')==2
    AA = imread(img_path);
    if size(AA,3)>1, AA = rgb2gray(AA); end
    AA = im2double(AA);
    [H,W] = size(AA); side=min(H,W);
    y1=floor((H-side)/2)+1; y2=y1+side-1;
    x1=floor((W-side)/2)+1; x2=x1+side-1;
    AA_sq   = AA(y1:y2, x1:x2);
    AA_disk = imresize(AA_sq, [2*R+1, 2*R+1], 'bilinear');
    f = zeros(S,S);
    f(ctr-R:ctr+R, ctr-R:ctr+R) = AA_disk;
else
    warning('Image not found. Using a synthetic smooth test pattern.');
    f = zeros(S,S);
    [Xg,Yg] = meshgrid(1:S,1:S);
    rr=hypot(Xg-ctr,Yg-ctr)/R; ang=atan2(Yg-ctr,Xg-ctr);
    f = exp(-rr.^2).*(0.4 + 0.6*cos(3*ang));
end
f(~inside)=0; f(inside)=mat2gray(f(inside));
end

%% ======================= Pairwise diagnostics (UNITARY) ==============
function out = print_pairwise_M2_diagnostics_unit(M2pairs_red, M2pairs_true, name)
% Inputs: [nr x nr x Nphi] each (unitary φ-domain)
nr = size(M2pairs_true,1);
rel = zeros(nr,nr);
als = zeros(nr,nr);
cr  = zeros(nr,nr);

for i=1:nr
  for j=1:nr
    A = double(squeeze(M2pairs_red(i,j,:)));
    B = double(squeeze(M2pairs_true(i,j,:)));
    rel(i,j) = norm(A(:)-B(:))/max(norm(B(:)),1e-30);
    a=B(:); b=A(:); als(i,j)=(a'*b)/max(a'*a,1e-30);
    a0=a-mean(a); b0=b-mean(b); cr(i,j)=(a0'*b0)/max(norm(a0)*norm(b0),1e-30);
  end
end

fprintf('\n%s diagnostics over all (i,j):\n', name);
fprintf('  rel_err: mean=%.3e  median=%.3e  max=%.3e\n', mean(rel(:)), median(rel(:)), max(rel(:)));
fprintf('  alpha  : mean=%.4f  median=%.4f\n', mean(als(:)), median(als(:)));
fprintf('  corr   : mean=%.4f  median=%.4f\n', mean(cr(:)),  median(cr(:)));

out = struct();
out.pairs = struct('mean_rel',mean(rel(:)),'median_rel',median(rel(:)),'max_rel',max(rel(:)));
end

%% -------------------- mini utilities for the summary lines ----------
% ----- Dimension-aware relative error helpers (work for M3 and M2) -----
function s = per_ring_rel_errors(Ared, Atru)
% Returns a vector of per-ring relative errors.
% - If inputs are 3-D (e.g., M3: Nphi x Nphi x nr), it uses Frobenius per slice.
% - If inputs are 2-D (e.g., M2: Nphi x nr), it uses 2-norm per column.
if ndims(Atru) >= 3 && size(Atru,3) > 1
    nr = size(Atru,3);
    s  = zeros(nr,1);
    for r = 1:nr
        num = norm(Ared(:,:,r) - Atru(:,:,r), 'fro');
        den = max(norm(Atru(:,:,r), 'fro'), 1e-30);
        s(r) = num / den;
    end
else
    nr = size(Atru,2);
    s  = zeros(nr,1);
    for r = 1:nr
        num = norm(Ared(:,r) - Atru(:,r));
        den = max(norm(Atru(:,r)), 1e-30);
        s(r) = num / den;
    end
end
end

function m = mean_rel_err(Ared, Atru)
s = per_ring_rel_errors(Ared, Atru);
m = mean(s);
end

function m = median_rel_err(Ared, Atru)
s = per_ring_rel_errors(Ared, Atru);
m = median(s);
end

function m = max_rel_err(Ared, Atru)
s = per_ring_rel_errors(Ared, Atru);
m = max(s);
end
function theta_ref = refine_angles_by_band_cc(rings_rot, theta_in, bands, Nphi, ref_mode)
% rings_rot: R x Nphi (current, BEFORE applying theta_in)
% theta_in : R x 1   (radians) from your spectral sync
% bands    : cell array of vectors of ring indices (1-based)
% ref_mode : 'self' (vs. global) — see below
R = size(rings_rot,1);
theta_ref = theta_in;

% rotate rings by current theta
rotRings = zeros(size(rings_rot));
for r=1:R
    rotRings(r,:) = apply_fractional_rotation_ring_nonunit(rings_rot(r,:), theta_in(r), Nphi);
end

% build band templates (simple mean; you can median)
Band = cell(numel(bands),1);
for b=1:numel(bands)
    idx = bands{b};
    Band{b} = mean(rotRings(idx,:), 1);   % 1 x Nphi
end

% choose reference: 'self' uses adjacent-band consistency;
% 'image' compares to image-derived rings (if you have them)
for b=1:numel(bands)
    if strcmp(ref_mode,'self')
        % align band b to average of neighbors (if exist)
        nbr = [];
        if b>1 
            nbr = [nbr, Band{b-1}]; 
        end
        if b<numel(bands) 
            nbr = [nbr, Band{b+1}]; 
        end
        if isempty(nbr)
            continue; 
        end
        target = mean(reshape(nbr,[],size(Band{b},2)),1);
    else
        % or: use reduction truth for intra-band M2 to synthesize target band
        target = Band{b}; % placeholder if no external reference
    end

    % phase correlation on the circle (subsampled helps robustness)
    s = est_circular_shift(Band{b}, target);   % radians
    % apply the same offset to all rings in the band
    theta_ref(bands{b}) = wrapToPi(theta_ref(bands{b}) + s);
end
end

function s = est_circular_shift(x,y)
% return rotation (radians) s that best aligns x(φ) ≈ y(φ+s)
N = numel(x);
X = fft(x); Y = fft(y);
xc = ifft(conj(Y).*X); [~,kmax] = max(real(xc));
shift_samp = kmax - 1;                   % samples
s = 2*pi * shift_samp / N;               % radians
end

function theta_sm = smooth_angles_graph_radial(theta_init, edges, dtheta_ij, w, R, lambda)
% Fix gauge theta(1)=0
theta = theta_init(:);
% Build normal matrix A and rhs b for edge residuals
I = edges(:,1); J = edges(:,2);
m = numel(w);
% incidence for edges
Ee = sparse([(1:m)';(1:m)'], [I;J], [ones(m,1);-ones(m,1)], m, R);
W = spdiags(w,0,m,m);
A1 = Ee' * W * Ee;
b1 = Ee' * (W * dtheta_ij(:));

% second-difference smoothness
e = ones(R,1);
D2 = spdiags([e -2*e e], -1:1, R, R);
D2(1,:)=0; D2(R,:)=0;           % natural boundary
A = A1 + lambda * (D2'*D2);
b = b1;

% gauge: pin node 1 to zero (or to theta_init(1))
A(1,:) = 0; A(:,1) = 0; A(1,1)=1; b(1)=0;

theta_sm = A \ b;
% wrap
theta_sm = angle(exp(1i*theta_sm));
end
