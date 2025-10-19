function pipeline_3D_top()
% PIPELINE_3D_TOP
% ------------------------------------------------------------------------------------
% End-to-end validator for SO(3) bispectrum–based coefficient recovery from a 3-D
% volume, including an SE(3)→SO(3) reduction via boundary-weighted averaging over
% translations. The script:
%
%   1) Loads a reference 3-D volume and masks/pads it.
%   2) Projects the volume onto spherical shells (Gaussian radial basis) and
%      spherical harmonics Y_ℓ^m up to Lmax, yielding per-shell SH blocks F{ℓ+1}(:,c).
%   3) Builds rotation invariants for the solver:
%        • G2 = F^H F  (degree-2 Gram blocks)
%        • Bisys (degree-3 linear systems) either:
%            – Oracle (exact rows from Gaunt ∘ kron(f1,f2) with Tabc), or
%            – Boundary-weighted SE(3) reduction (BW) via translation averaging.
%   4) Solves for SH coefficients per (ℓ,c) by regularized least squares.
%   5) Equalizes per-shell amplitudes (optional), synthesizes volumes, and reports
%      quality metrics (PSNR, SSIM, relL2, NMSE). Also prints diagnostics for
%      Gram errors and condition numbers.
%
% Key ideas for the BW reduction:
%   SO(3) moments are obtained as a δ→0 boundary limit of SE(3) autocorrelations:
%       ⟨•⟩_SO(3) = lim_{δ→0+}   E_t [ s_δ(t) · (•) ] / E_t [ s_δ(t) ],
%   where s_δ(t) = ∫_{S^2} f(t+r0θ) f(t−r0θ) dσ(θ), with r0 = R(1−δ).
%   Practically, we (i) enumerate translations t, (ii) compute s_δ(t),
%   (iii) project the translated volume V_t to F^(t), and (iv) accumulate
%   s_δ(t)·(G2^(t), Bisys^(t)) and normalize by Σ_t s_δ(t).
%
% OUTPUTS
%   (None returned) – the function prints diagnostics to the console and shows
%   reference and recovered volumes via volshow (if available).
%
% DEPENDENCIES (must be on path):
%   build_gaunt_tables_w3j, build_Ycache (below), volume_to_SH_coeffs_radial,
%   assemble_bisys_oracle, assemble_bisys_oracle_from_plan,
%   build_bisys_sampling_plan, build_BW_invariants_from_translations,
%   recover_so3_from_bispectrum, synth_volume_from_SH, sphharmY, fib_sphere, etc.
%
% ------------------------------------------------------------------------------------

% --------------------------- Reproducibility ---------------------------------------
rng(0);

% ------------------------------ Configuration --------------------------------------
d_init_vol = 128;   % Base cubic grid for the input volume before padding
Lmax       = 5;     % Max SH degree
R          = 16;    % # radial shells (Gaussian basis)
Nquad      = 384*16;% # S^2 nodes for projection (Fibonacci sphere)
eq_per_c   = 4000;  % Target rows per (ℓ,c) block for the bispectrum system
reg_lambda = 0;     % Solver Tikhonov regularization (per block)
shell_bw   = 0.040; % Gaussian width of shells (in r∈[0,1])

% Boundary-weighted (BW) reduction configuration
bw_cfg.enable           = true;   % Compute BW invariants (G2, Bisys)
bw_cfg.grid_n           = 3;      % Odd; builds grid_n^3 translation grid
bw_cfg.step_vox         = 2;      % Translation step in voxels
bw_cfg.pad_vox          = 2;      % Zero padding (voxels) around base volume
bw_cfg.mask_rad         = 0.98;   % Sphere mask radius inside [-1,1]^3
bw_cfg.delta_boundary   = 1e-2;   % δ for s_δ(t) boundary factor (1e-3..1e-2)
bw_cfg.rows_per_c       = eq_per_c;
bw_cfg.row_normalize    = true;   % Normalize rows AFTER BW averaging
bw_cfg.seed             = 0;      % Fixed plan reproducibility
bw_cfg.verbose          = 1;

% Use BW invariants in the solver? (if false, use oracle invariants)
use_bw_for_solver = bw_cfg.enable;

% Effective working grid after padding
d = d_init_vol + 2*bw_cfg.pad_vox;

% ----------------------------- Data paths / choices --------------------------------
use_ref_x = false;
% --- Convert relative paths to absolute paths ---
thisFile  = mfilename('fullpath');    % full path of current .m file
thisDir   = fileparts(thisFile);      % directory of this file
repoRoot  = fileparts(thisDir);       % one level up (adjust if needed)
assetsDir = fullfile(repoRoot, 'assets');  % ../assets

% Build full absolute paths
pth_x = fullfile(assetsDir, 'emdb_2984.mat');
pth_y = fullfile(assetsDir, 'S80_ribosome.mat');

fprintf('[CFG] d=%d | Lmax=%d | R=%d | Nquad=%d | eq/target≈%d | shell_bw=%.3f \n', ...
        d, Lmax, R, Nquad, eq_per_c, shell_bw);

% --------------------------------- Load volumes ------------------------------------
fprintf('[LOAD] Loading volumes ...\n');
emdb_2984 = load(pth_x);
emdb_2660 = load(pth_y);

vol_x = squeeze(emdb_2984.volume(1,:,:,:));
vol_x = imresize3(double(vol_x), [d_init_vol, d_init_vol, d_init_vol]);
vol_x = imrotate3(vol_x, 90, [0,1,1], 'cubic', 'crop');
vol_x = vol_x / max(norm(vol_x(:), 'fro'), eps);

vol_y = squeeze(emdb_2660.original_vol(1,31:236,31:236,31:236));
vol_y = imresize3(double(vol_y), [d_init_vol, d_init_vol, d_init_vol]);
vol_y = imrotate3(vol_y, 300, [0,0,1], 'cubic', 'crop');
vol_y = vol_y / max(norm(vol_y(:), 'fro'), eps);

if use_ref_x
    Vref = vol_x; refname = 'emdb_2984';
else
    Vref = vol_y; refname = 'S80_ribosome';
end

% Mask inside a sphere then pad for the translation grid
Vref = apply_spherical_mask(Vref, bw_cfg.mask_rad);
Vref = padarray(Vref, [bw_cfg.pad_vox, bw_cfg.pad_vox, bw_cfg.pad_vox], 0, 'both');
fprintf('  Using reference volume: %s\n', refname);

% ---------------------------- S^2 nodes and Gaunt tables ---------------------------
[th, ph, w_ang] = fib_sphere(Nquad);                  % S^2 quadrature
fprintf('[GAUNT] Building Gaunt tables (to Lmax=%d) ... ', Lmax);
tic; Gaunt = build_gaunt_tables_w3j(Lmax); fprintf('done (%.2fs).\n', toc);

Ycache   = build_Ycache(Lmax, th, ph);                % Y_ℓ^m(θk,φk) cache
r_shells = linspace(0.12, 0.96, R);                   % Shell centers

% --------------------- Volume → shells × SH (Ftrue, Tabc) --------------------------
fprintf('[ANALYSIS] Projecting reference volume onto R=%d shells and Lmax=%d ...\n', R, Lmax);
[Ftrue, Tabc, shell_cnorm] = volume_to_SH_coeffs_radial( ...
    Vref, Lmax, r_shells, th, ph, w_ang, Ycache, shell_bw);

% Oracle G2 for diagnostics
G2_true = cell(Lmax+1,1);
for ell = 0:Lmax
    G2_true{ell+1} = Ftrue{ell+1}' * Ftrue{ell+1};
end

% ------------------------- Oracle invariants (for comparison) -----------------------
fprintf('[INV:M2] Using ORACLE G2 (Ftrue^H Ftrue).\n');
G2_oracle = G2_true;
print_G2_diagnostics(G2_oracle, G2_true, Lmax);

fprintf('[INV:M3] Using ORACLE assembly (Gaunt + Tabc + Ftrue) for Bisys.\n');
[Bisys_oracle, ~] = assemble_bisys_oracle(Ftrue, Gaunt, Tabc, Lmax, R, eq_per_c);

% ============================= BW Reduction (SE(3)→SO(3)) ==========================
fprintf('[REDUCTION] Building BW sampling plan & evaluating BW invariants ...\n');

% 1) Translations on a symmetric voxel grid
Tvox = make_translation_grid_vox(bw_cfg.grid_n, bw_cfg.step_vox);

% 2) Deterministic sampling plan (reused for all translations)
plan = build_bisys_sampling_plan(Lmax, R, Tabc, Gaunt, ...
                                 bw_cfg.rows_per_c, bw_cfg.seed, bw_cfg.verbose);

% 3) Accumulate boundary-weighted invariants across translations
[G2_BW, Bisys_BW] = build_BW_invariants_from_translations( ...
    Vref, Lmax, R, r_shells, shell_bw, th, ph, w_ang, Ycache, ...
    Gaunt, Tabc, plan, Tvox, bw_cfg.delta_boundary, bw_cfg.mask_rad, ...
    bw_cfg.row_normalize, bw_cfg.verbose);

% --------------------- Compare BW vs Oracle (same plan for Bisys) -------------------
fprintf('[COMPARE] BW G2 vs ORACLE G2 (relative Frobenius per ℓ)\n');
print_G2_diagnostics(G2_BW, G2_true, Lmax);

fprintf('[COMPARE] BW Bisys vs ORACLE (with SAME PLAN) per (ℓ3,c)\n');
Bisys_oracle_plan = assemble_bisys_oracle_from_plan(Ftrue, Gaunt, Tabc, plan, ...
                                                    bw_cfg.row_normalize, Lmax, R);
print_Bisys_diagnostics(Bisys_BW, Bisys_oracle_plan, Lmax, R);

% ------------------------------ Pick invariants to solve ----------------------------
if use_bw_for_solver
    fprintf('[INV] Using BW invariants for SOLVE.\n');
    G2_in = G2_BW;  Bisys = Bisys_BW;
else
    fprintf('[INV] Using ORACLE invariants for SOLVE.\n');
    G2_in = G2_oracle;  Bisys = Bisys_oracle;
end

% ------------------------------------- Solve ---------------------------------------
invData = struct();
invData.Lmax       = Lmax;
invData.R          = R;
invData.mL         = arrayfun(@(l) 2*l+1, 0:Lmax);
invData.Gaunt      = Gaunt;
invData.G2         = G2_in;
invData.Bisys      = Bisys;
invData.F_l1_known = Ftrue{1+1}; % optional gauge on ℓ=1

opts = struct('verbose', 1, 'reg_lambda', reg_lambda, ...
              'min_eq_per_col', 3, 'condition_warn', 1e9);

fprintf('[SOLVE] Recovering coefficients ...\n');
[Fhat, dout] = recover_so3_from_bispectrum(invData, opts);

% ---------------------------- Coefficient diagnostics ------------------------------
fprintf('[EVAL:COEFF] Per-ℓ Procrustes error and Gram mismatch.\n');
perEll_err = zeros(Lmax+1,1);
for ell = 0:Lmax
    X = Fhat{ell+1}; Y = Ftrue{ell+1};
    [Uq,~,Vq] = svd(Y*X','econ'); Qg = Uq*Vq'; Xal = Qg*X;
    rel = norm(Xal - Y,'fro') / max(norm(Y,'fro'),eps);
    perEll_err(ell+1) = rel;
    conds = arrayfun(@(c) safeget(dout.perEll{ell+1}, c, 'condA', NaN), 1:R);
    fprintf('  [ℓ=%d] relFrobErr=%.3e | median cond(A)=%.2e | GramRelErr=%.3e\n', ...
            ell, rel, nanmedian(conds), dout.G2_rel_err(ell+1));
end
fprintf('  ==> max per-ℓ coeff error = %.3e\n', max(perEll_err));

% ----------------------------- Synthesis & metrics ---------------------------------
fprintf('[SYNTH] Synthesizing recovered 3-D volume on %d^3 grid ...\n', d);
mask_rad = 0.95;

Vrec    = synth_volume_from_SH(Fhat,  Lmax, r_shells, shell_bw, d, mask_rad, shell_cnorm);
Voracle = synth_volume_from_SH(Ftrue, Lmax, r_shells, shell_bw, d, mask_rad, shell_cnorm);

Vref_norm = normalize_volume_for_metrics(Vref,   mask_rad);
Vrec_norm = normalize_volume_for_metrics(Vrec,   mask_rad);
Vor_norm  = normalize_volume_for_metrics(Voracle,mask_rad);

[PSNRdB, SSIMv, relL2, NMSE] = volume_metrics(Vref_norm, Vrec_norm);
[PSNR_or, SSIM_or, ~, ~]     = volume_metrics(Vref_norm, Vor_norm);

fprintf('[METRICS:VOLUME] PSNR=%.2f dB | SSIM=%.4f | relL2=%.3e | NMSE=%.3e\n', ...
        PSNRdB, SSIMv, relL2, NMSE);
fprintf('[CEILING] Oracle synth from Ftrue: PSNR=%.2f dB | SSIM=%.4f\n', PSNR_or, SSIM_or);

% (Optional) rough global rotation estimate from ℓ=1
try
    Rglob = estimate_global_rotation_from_l1(Fhat{1+1}, Ftrue{1+1}); 
catch
end

fprintf('[DEBUG] voxel stats: ref[min,med,max]=[%.3e %.3e %.3e], rec[min,med,max]=[%.3e %.3e %.3e]\n', ...
        min(Vref_norm(:)), median(Vref_norm(:)), max(Vref_norm(:)), ...
        min(Vrec_norm(:)),  median(Vrec_norm(:)),  max(Vrec_norm(:)));

% -------------------------------- Visualization ------------------------------------
try
    figure('Name','Reference volume (volshow)');  
    volshow(Vref_norm,'RenderingStyle','MaximumIntensityProjection');
    figure('Name','Recovered volume (volshow)');
    volshow(Vrec_norm,'RenderingStyle','MaximumIntensityProjection');
catch
    % volshow may be unavailable (e.g., no Image Processing Toolbox / headless)
end
end


% ================================== Helpers ========================================

function v = safeget(S, idx, fld, def)
% SAFEGET  Defensive getter for struct arrays.
if idx <= numel(S) && isfield(S(idx),fld) && ~isempty(S(idx).(fld))
    v = S(idx).(fld);
else
    v = def;
end
end

function Vout = apply_spherical_mask(Vin, radius)
% APPLY_SPHERICAL_MASK  Zero out voxels outside a centered sphere of given radius.
% radius ∈ (0,1]; grid is assumed on [-1,1]^3.
d = size(Vin,1);
[x,y,z] = ndgrid(linspace(-1,1,d), linspace(-1,1,d), linspace(-1,1,d));
mask = (x.^2 + y.^2 + z.^2) <= radius^2;
Vout = Vin;
Vout(~mask) = 0;
end

function Tvox = make_translation_grid_vox(grid_n, step_vox)
% MAKE_TRANSLATION_GRID_VOX  Symmetric voxel translation grid centered at 0.
% grid_n must be odd: e.g., 3 → {-step, 0, +step} in each axis.
if mod(grid_n,2)==0, error('grid_n must be odd'); end
r = (-(grid_n-1)/2:(grid_n-1)/2) * step_vox;
[TZ, TY, TX] = ndgrid(r, r, r);  % MATLAB’s volume order: z,y,x
Tvox = [TX(:), TY(:), TZ(:)];    % rows: [dx, dy, dz] in voxels
end

function [theta, phi, w] = fib_sphere(N)
% FIB_SPHERE  Approximately uniform S^2 nodes (Fibonacci), with equal weights.
g = (sqrt(5)-1)/2;
k = (0:N-1)'; z = 1 - 2*(k+1/2)/N;
theta = acos(max(min(z,1),-1));        % [0,π]
phi   = 2*pi*mod(k*g, 1);               % [0,2π)
w     = (4*pi)/N * ones(N,1);
end

function Ycache = build_Ycache(Lmax, th, ph)
% BUILD_YCACHE  Cache Y_ℓ^m(θ,φ) per ℓ with rows m=−ℓ:ℓ and columns angle nodes.
N = numel(th);
Ycache = cell(Lmax+1,1);
for l = 0:Lmax
    m = 2*l+1; ms = (-l:l).';
    Ylm = zeros(m, N);
    for j=1:m, Ylm(j,:) = sphharmY(l, ms(j), th, ph); end
    Ycache{l+1} = Ylm;
end
end

function R = estimate_global_rotation_from_l1(Fhat_l1, Ftrue_l1)
% ESTIMATE_GLOBAL_ROTATION_FROM_L1  Procrustes in the real ℓ=1 (vector) subspace.
X = complex_l1_to_real3(Fhat_l1);
Y = complex_l1_to_real3(Ftrue_l1);
[U,~,V] = svd(Y*X','econ'); R = U*V';
if det(R)<0, U(:,3) = -U(:,3); R = U*V'; end
end

function M = complex_l1_to_real3(F1)
% COMPLEX_L1_TO_REAL3  Map complex ℓ=1 SH coeffs → real 3-vector representation.
F = zeros(3,size(F1,2));
F(1,:) = real( (F1(1,:) - F1(3,:)) / (sqrt(2)*1i) );
F(2,:) = real( F1(2,:) );
F(3,:) = real( (F1(1,:) + F1(3,:)) / sqrt(2) );
M = F;
end

function Vn = normalize_volume_for_metrics(V, mask_rad)
% NORMALIZE_VOLUME_FOR_METRICS  Zero-mean/unit-RMS inside spherical mask.
N = size(V,1);
[x,y,z] = ndgrid(linspace(-1,1,N));
mask = sqrt(x.^2+y.^2+z.^2) <= mask_rad;
v = V(mask);
v = v(~isnan(v));
v = v - mean(v(:));
rmsv = sqrt(mean(v(:).^2)); if rmsv>0, v = v/rmsv; end
Vn = nan(size(V)); Vn(mask) = v;
end

function [PSNRdB, SSIMv, relL2, NMSE] = volume_metrics(Vref, Vrec)
% VOLUME_METRICS  Simple PSNR/SSIM-like measures on masked voxels.
mask = ~isnan(Vref) & ~isnan(Vrec);
X = Vref(mask); Y = Vrec(mask);
alpha = (Y'*X) / max(Y'*Y, eps);
Yal   = alpha * Y;
relL2 = norm(Yal - X) / max(norm(X),eps);
NMSE  = mean(abs(Yal - X).^2) / max(mean(abs(X).^2),eps);
peak  = max(abs(X)); mse = mean(abs(Yal - X).^2);
PSNRdB = 10*log10( max(peak^2/mse, eps) );
SSIMv = ssim3d_like(embed_mask(X,Vref), embed_mask(Yal,Vref));
end

function V = embed_mask(vec, Vtmpl)
mask = ~isnan(Vtmpl);
V = nan(size(Vtmpl));
V(mask) = vec;
end

function s = ssim3d_like(A,B)
s = mean([ssim_slice(A,B,3), ssim_slice(A,B,2), ssim_slice(A,B,1)]);
end

function v = ssim_slice(A,B,dim)
sz = size(A); mid = ceil(sz(dim)/2);
switch dim
  case 3, A2 = squeeze(A(:,:,mid)); B2 = squeeze(B(:,:,mid));
  case 2, A2 = squeeze(A(:,mid,:)); B2 = squeeze(B(:,mid,:));
  case 1, A2 = squeeze(A(mid,:,:)); B2 = squeeze(B(mid,:,:));
end
A2 = normalize2d(A2); B2 = normalize2d(B2);
try
    v = ssim(A2,B2);
catch
    e = norm(A2(:)-B2(:))/max(norm(A2(:)),eps); v = max(0, 1 - e);
end
end

function X = normalize2d(X)
X(isnan(X)) = 0;
X = X - mean(X(:));
sd = std(X(:)); if sd>0, X = X/sd; end
end

function print_G2_diagnostics(G2_est, G2_true, Lmax)
% PRINT_G2_DIAGNOSTICS  Reports per-ℓ relative Frobenius error of Gram blocks.
fprintf('[DEBUG:M2->G2] Per-ℓ ||G2_est - G2_true||_F / ||G2_true||_F:\n');
for ell=0:Lmax
    Ge = G2_est{ell+1}; Gt = G2_true{ell+1};
    rel = norm(Ge - Gt, 'fro') / max(norm(Gt,'fro'), eps);
    fprintf('  [ℓ=%d] rel=%.3e\n', ell, rel);
end
end

function print_Bisys_diagnostics(BW, OR, Lmax, R)
% PRINT_BISYS_DIAGNOSTICS  Compare BW vs Oracle-with-same-plan per (ℓ3,c).
for ell=0:Lmax
  for c=1:R
    A1 = BW(ell+1,c).A; b1 = BW(ell+1,c).b;
    A2 = OR(ell+1,c).A; b2 = OR(ell+1,c).b;
    if isempty(A1) || isempty(A2)
        fprintf('  [Bisys][ℓ=%d,c=%d] rows: BW=%d OR=%d (skip)\n', ell,c,size(A1,1),size(A2,1));
        continue;
    end
    n = min(size(A1,1), size(A2,1));
    A1c = A1(1:n,:); A2c = A2(1:n,:);
    b1c = b1(1:n);   b2c = b2(1:n);
    relA = norm(A1c - A2c, 'fro') / max(norm(A2c, 'fro'), eps);
    relb = norm(b1c - b2c) / max(norm(b2c), eps);
    fprintf('  [Bisys][ℓ=%d,c=%d] rows=%d | relFrob(A)=%.3e | relL2(b)=%.3e\n', ...
            ell, c, n, relA, relb);
  end
end
end
