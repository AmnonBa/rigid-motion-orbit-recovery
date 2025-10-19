% MAIN_STATISTICAL_M2M3_REDUCTION
% ---------------------------------------------------------------
% Validates the single-value SE(2)->SO(2) reduction for:
%   • M^2 at (r1, r2, phi)
%   • M^3 at (r1, r2, r3, dphi1, dphi2)
% For noisy averaging convergence (ratio-of-sums) - fo varous SNR levels 
% number of observations.
% ---------------------------------------------------------------

rng(1);

%% ---------------- Test image (disk-supported) -----------------

Rdisk = 30;                 % disk radius in pixels
S     = 2*Rdisk + 11;                 % padded canvas size (odd)
[X,Y] = meshgrid(1:S,1:S);
ctr   = (S+1)/2;
R     = hypot(X-ctr, Y-ctr);
mask  = (R <= Rdisk);

% Simple anisotropic Gaussian mixture inside the disk
f = zeros(S);
f = f + 1.3*gauss2d(S, ctr+[-25,+10], 10, 15, 0.2);
f = f + 0.9*gauss2d(S, ctr+[+20,-30], 18, 10, -0.7);
f = f + 0.7*gauss2d(S, ctr+[+0,+0],   30, 30, 0.0);
f(~mask) = 0;
%f = f - mean(f(mask));       % zero-mean inside disk (nice numerically)

%% ---------------- Config (matches your v5) --------------------
cfg = struct();
cfg.R        = Rdisk;
cfg.pad      = 4;
cfg.Nphi     = 180;          % angular harmonics per ring
cfg.Nphi_os  = [];          % optional oversampling for ring sampling
cfg.delta_pix= 0.05;          % boundary inset
cfg.margin_ring = 0.5;       % ring margin (pixels)
cfg.ringMethod = 'cubic';
cfg.bndMethod  = 'cubic';
cfg.limit_translations = true;
cfg.trans_radius       = 1;   % translations near center
cfg.Ntheta             = max(180, round(4*pi*Rdisk)); % boundary directions
cfg.y_chunk            = 512;
cfg.useGPU             = true;   % toggle true/false

%% Loading optional images if desired
if false
% --- Build relative path to assets/ and choose image ---
thisFile  = mfilename('fullpath');           % full path of current .m file
thisDir   = fileparts(thisFile);             % folder containing this file
repoRoot  = fileparts(thisDir);              % one level up (adjust if needed)
assetsDir = fullfile(repoRoot, 'assets');    % ../assets

imgName   = 'albert-einstein.jpg';                  % change to 'monaLiza.jpg' if you like
img_path  = fullfile(assetsDir, imgName);

if ~exist(img_path, 'file')
    error('Image not found at: %s', img_path);
end

% --- Load, embed on disk, and report ---
[f, inside] = load_embed_disk_image_local(img_path, cfg.R, cfg.pad);
S = size(f,1);
ctr = (S+1)/2;
mask = disk_mask(S, cfg.R);

fprintf('Loaded %s\nCanvas size %dx%d, disk radius R=%d (pad=%d)\n', ...
        imgName, S, S, cfg.R, cfg.pad);
end

%% ---------------- Targets (grid-aligned for exact IFFT baseline) ----
kphi   = 17;                             % grid index for phi
phi    = 2*pi*(kphi)/cfg.Nphi;          % exact grid angle
kphi1  = 45; kphi2 = 67;               % grid indices for dphi1,dphi2
dphi1  = 2*pi*(kphi1)/cfg.Nphi;
dphi2  = 2*pi*(kphi2)/cfg.Nphi;

r1 = 12; r2 = 14; r3 = 16;              % radii (0-based)

target = struct();
target.M2 = struct('r1', r1, 'r2', r2, 'phi',  phi);
target.M3 = struct('r1', r1, 'r2', r2, 'r3', r3, 'dphi1', dphi1, 'dphi2', dphi2);

%% ---------------- Single-value reduction ----------------------
%tic;
out = se2_to_so2_M2M3_single_batched(f, cfg, target);
%toc;
out.M2_val = out.M2_val / out.D_scalar;
out.M3_val = out.M3_val / out.D_scalar;

fprintf('Single-value reduction:\n');
fprintf('  D_scalar = %.6e\n', out.D_scalar);
fprintf('  M2_red(r1=%d,r2=%d;phi=%.4f)  = %.12e\n', r1, r2, phi,  out.M2_val);
fprintf('  M3_red(r1=%d,r2=%d,r3=%d;d1=%.4f,d2=%.4f) = %.12e\n', r1, r2, r3, dphi1, dphi2, out.M3_val);
%tic;
out2 = se2_to_so2_M2M3_single(f, cfg, target);


%% ---------------- Baseline via grid IFFT (sanity) -------------
[prep, rings] = build_minimal_pieces(f, cfg, [r1 r2 r3]);

% M^2 grid baseline
C1  = rings{r1+1};
C2n = rings{r2+1}(prep.IDX1D,:);
Sspec = (C1 .* C2n) * prep.sa;                        % [Nphi x 1]
M2_col_grid = real(ifft(Sspec,[],1));            % ifft already has 1/Nphi
M2_grid_val = M2_col_grid(kphi+1) / out.D_scalar;

% M^3 grid baseline
Bsum = zeros(cfg.Nphi, cfg.Nphi);
Ny_act = numel(prep.sa);
for y0 = 1:cfg.y_chunk:Ny_act
    y1 = min(Ny_act, y0 + cfg.y_chunk - 1);
    yc = y0:y1;  K = numel(yc);
    A  = reshape(rings{r1+1}(:,yc), cfg.Nphi, 1, K);
    B  = reshape(rings{r2+1}(:,yc), 1, cfg.Nphi, K);
    OP = pagemtimes(A, B);
    C3neg = reshape(rings{r3+1}(prep.IDX2D_flat, yc), cfg.Nphi, cfg.Nphi, K);
    S  = reshape(prep.sa(yc), 1,1,K);
    Bsum = Bsum + sum(OP .* C3neg .* S, 3);
end
M3_plane_grid = real(ifft2(Bsum)) * cfg.Nphi/sqrt(cfg.Nphi);
M3_grid_val   = M3_plane_grid(kphi1+1, kphi2+1) / out.D_scalar;

fprintf('\nGrid-FFT baselines (exact on-grid):\n');
fprintf('  M2_grid  = %.12e   | abs diff = %.3e\n', M2_grid_val, abs(M2_grid_val - out.M2_val));
fprintf('  M3_grid  = %.12e   | abs diff = %.3e\n', M3_grid_val, abs(M3_grid_val - out.M3_val));

%% ---------------- TRUE SO(2) moments (direct rotation average) --
% We evaluate the definitions:
%   M2_true = E_theta[ f(r1,-theta) * f(r2, phi - theta) ]
%   M3_true = E_theta[ f(r1,-theta) * f(r2, dphi1 - theta) * f(r3, dphi2 - theta) ]
% using pre-sampled ring values of f at the origin (no translations).

H1 = ring_values_one(f, r1, cfg);           % length Nphi, samples at angles 2πj/Nphi
H2 = ring_values_one(f, r2, cfg);
H3 = ring_values_one(f, r3, cfg);

Nphi = cfg.Nphi;
idx  = 0:Nphi-1;                             % theta_j = 2π j/Nphi

% grid-aligned indices for the relative angles
i_neg = mod(-idx, Nphi) + 1;                 % -theta_j
i_p   = mod(kphi  - idx, Nphi) + 1;          % phi - theta_j
i_p1  = mod(kphi1 - idx, Nphi) + 1;          % dphi1 - theta_j
i_p2  = mod(kphi2 - idx, Nphi) + 1;          % dphi2 - theta_j

M2_true = mean( H1(i_neg) .* H2(i_p) );
M3_true = mean( H1(i_neg) .* H2(i_p1) .* H3(i_p2) );

fprintf('\nTRUE SO(2) moments (direct rotation average of f):\n');
fprintf('  M2_true  = %.12e   | abs diff (red-true) = %.3e  | rel = %.3e\n', ...
        M2_true, abs(out.M2_val - M2_true), relerr(out2.M2_val, M2_true));
fprintf('  M3_true  = %.12e   | abs diff (red-true) = %.3e  | rel = %.3e\n', ...
        M3_true, abs(out.M3_val - M3_true), relerr(out2.M3_val, M3_true));

%% ================== Experiment grid ==================
% Tune these:
snr_factors = [0.25 0.5 1 2 3 4 6 8];         % sigma = snr_factor * std(f(mask))
M_list      = round(logspace(log10(1e4), log10(1e5), 12));   % number of Monte Carlo samples per trial
nTrials     = 300;              % repetitions per (sigma, M)
baseSeed    = 123;             % reproducible randomness

% Pick which slices to plot:
plot_M_fixed_idx    = numel(M_list);      % use the largest M for SNR-sweep
plot_sigma_fixed_idx= numel(snr_factors); % use the largest noise for M-sweep

% Precompute base noise scale:
sigma_base = std(f(mask));

% Storage: per-trial estimates and rel errors (vs TRUE)
nS = numel(snr_factors); nM = numel(M_list);

%% ================== Main sweeps ==================
% Pre-alloc (same shapes as before)
% ----- Prealloc: TRIALS first, then M, then SIGMA -----
M2_est  = nan(nTrials, nM, nS);
M3_est  = nan(nTrials, nM, nS);
M2_rel  = nan(nTrials, nM, nS);
M3_rel  = nan(nTrials, nM, nS);
M2_rel2 = nan(nTrials, nM, nS);
M3_rel2 = nan(nTrials, nM, nS);

Mmax = max(M_list(:));

% You can switch to parfor t=1:nTrials if you like
for t = 1:nTrials
    fprintf('Trial %d/%d: generating up to Mmax = %d ...\n', t, nTrials, Mmax);

    for si = 1:nS
        sigma  = snr_factors(si) * sigma_base;
        fprintf('SNR values %d: ...\n', si);
        tic;
        % Trial-specific RNG stream (independent of mi/M)
        seed   = baseSeed + 1000*si + t;
        stream = RandStream('Threefry','Seed',seed); 
        RandStream.setGlobalStream(stream);

        % Accumulative run: returns prefix sequences for k = 1..Mmax
        [~, ~, m2_seq, m3_seq] = run_one_trial_ratio_of_sums( ...
            f, cfg, target, Mmax, sigma, 'BatchSize', 1024, 'Antithetic', false);

        toc;
        % Read off results for each requested M (vectorized)
        m_idx          = M_list(:)';                 % 1 x nM
        M2_t_row       = m2_seq(m_idx);             % 1 x nM
        M3_t_row       = m3_seq(m_idx);             % 1 x nM

        % Errors vs baselines
        M2_e_row  = arrayfun(@(x) relerr(x, out.M2_val), M2_t_row);
        M3_e_row  = arrayfun(@(x) relerr(x, out.M3_val), M3_t_row);
        M2_e2_row = arrayfun(@(x) relerr(x, M2_true),    M2_t_row);
        M3_e2_row = arrayfun(@(x) relerr(x, M3_true),    M3_t_row);

        % Store into the 3-D tensors at (t,:,si) 
        M2_est(t,:,si)  = M2_t_row;
        M3_est(t,:,si)  = M3_t_row;
        M2_rel(t,:,si)  = M2_e_row;
        M3_rel(t,:,si)  = M3_e_row;
        M2_rel2(t,:,si) = M2_e2_row;
        M3_rel2(t,:,si) = M3_e2_row;
    end

    save('M2_M3_err_Oct_03_v2.mat',  "-v7.3");

    % (Optional) progress per M
    for mi = 1:nM
        %fprintf('   M = %d done (accumulative)\n', M_list(mi));
    end
end

%% ================== Summary (mean / median / percentiles across TRIALS) ==================
% Arrays are now sized (nTrials x nM x nS). Reduce along dim=1 (trials),
% then transpose so outputs are (nS x nM) for plotting.

M2_rel_mean = squeeze(mean(M2_rel, 1, 'omitnan')).';   % (nM x nS) -> (nS x nM)
M3_rel_mean = squeeze(mean(M3_rel, 1, 'omitnan')).';
M2_rel_med  = squeeze(median(M2_rel, 1, 'omitnan')).';
M3_rel_med  = squeeze(median(M3_rel, 1, 'omitnan')).';

% Optional: percentile bands across trials (use whatever p you like)
p_lo = 5;  p_hi = 95;
M2_rel_p05 = squeeze(prctile(M2_rel, p_lo, 1)).';       % (nS x nM)
M2_rel_p95 = squeeze(prctile(M2_rel, p_hi, 1)).';
M3_rel_p05 = squeeze(prctile(M3_rel, p_lo, 1)).';
M3_rel_p95 = squeeze(prctile(M3_rel, p_hi, 1)).';


%% ================== Plots ==================
outdir = fullfile(pwd, 'fig_out');
if ~exist(outdir, 'dir'), mkdir(outdir); end

%============ (A) Vary SNR at fixed M ============
mi = plot_M_fixed_idx;

figA = figure('Name','Rel error vs SNR (fixed M)');

% ----- M^2 -----
subplot(1,2,1);
xS = snr_factors(:);

% use median across trials for fitting (more robust)
y_med  = M2_rel_med(:,mi);
y_mean = M2_rel_mean(:,mi);

% fit in log-log (error vs sigma)
[C_med,  alpha_med,  yfit_med]  = fit_powerlaw_and_curve(xS, y_med);
[C_mean, alpha_mean, yfit_mean] = fit_powerlaw_and_curve(xS, y_mean);

loglog(xS, y_mean, '-o', 'LineWidth',1.5); hold on;
loglog(xS, y_med,  '--s','LineWidth',1.5);
loglog(xS, yfit_mean, ':', 'LineWidth',1.5);
loglog(xS, yfit_med,  '-.', 'LineWidth',1.5);
grid on;
xlabel('\sigma / std(f)'); ylabel('Rel. error (M^2)');
title(sprintf('M^2 @ M = %g', M_list(mi)));
set(gca,'XScale','log');

legend( ...
  'mean over trials', ...
  'median over trials', ...
  sprintf('fit mean: C=%.3g, \\alpha=%.3f', C_mean, alpha_mean), ...
  sprintf('fit median: C=%.3g, \\alpha=%.3f', C_med,  alpha_med), ...
  'Location','best');

% ----- M^3 -----
subplot(1,2,2);
y_med  = M3_rel_med(:,mi);
y_mean = M3_rel_mean(:,mi);

[C_med,  alpha_med,  yfit_med]  = fit_powerlaw_and_curve(xS, y_med);
[C_mean, alpha_mean, yfit_mean] = fit_powerlaw_and_curve(xS, y_mean);

loglog(xS, y_mean, '-o', 'LineWidth',1.5); hold on;
loglog(xS, y_med,  '--s','LineWidth',1.5);
loglog(xS, yfit_mean, ':', 'LineWidth',1.5);
loglog(xS, yfit_med,  '-.', 'LineWidth',1.5);
grid on;
xlabel('\sigma / std(f)'); ylabel('Rel. error (M^3)');
title(sprintf('M^3 @ M = %g', M_list(mi)));
set(gca,'XScale','log');

legend( ...
  'mean over trials', ...
  'median over trials', ...
  sprintf('fit mean: C=%.3g, \\alpha=%.3f', C_mean, alpha_mean), ...
  sprintf('fit median: C=%.3g, \\alpha=%.3f', C_med,  alpha_med), ...
  'Location','best');

% --- save Figure A ---
%exportgraphics(figA, fullfile(outdir, sprintf('relerr_vs_SNR_M%d.png', M_list(mi))), 'Resolution', 300);
%exportgraphics(figA, fullfile(outdir, sprintf('relerr_vs_SNR_M%d.pdf', M_list(mi))),  'ContentType','vector');
%savefig(figA, fullfile(outdir, sprintf('relerr_vs_SNR_M%d.fig', M_list(mi))));

%% ============ (B) Vary M at fixed SNR ============
si_fixed = plot_sigma_fixed_idx - 5;   % keep your original choice

figB = figure('Name','Rel error vs M (fixed SNR)');

% ----- M^2 -----
subplot(1,2,1);
xM = M_list(:);

y_med  = M2_rel_med(si_fixed,:).';
y_mean = M2_rel_mean(si_fixed,:).';

[C_med,  alpha_med,  yfit_med]  = fit_powerlaw_and_curve(xM, y_med);
[C_mean, alpha_mean, yfit_mean] = fit_powerlaw_and_curve(xM, y_mean);

loglog(xM, y_mean, '-o', 'LineWidth',1.5); hold on;
loglog(xM, y_med,  '--s','LineWidth',1.5);
loglog(xM, yfit_mean, ':', 'LineWidth',1.5);
loglog(xM, yfit_med,  '-.', 'LineWidth',1.5);
grid on;
xlabel('M'); ylabel('Rel. error (M^2)');
title(sprintf('M^2 @ \\sigma = %.3g', snr_factors(si_fixed)*sigma_base));
set(gca,'XScale','log');

legend( ...
  'mean over trials', ...
  'median over trials', ...
  sprintf('fit mean: C=%.3g, \\alpha=%.3f', C_mean, alpha_mean), ...
  sprintf('fit median: C=%.3g, \\alpha=%.3f', C_med,  alpha_med), ...
  'Location','best');

% ----- M^3 -----
subplot(1,2,2);
y_med  = M3_rel_med(si_fixed,:).';
y_mean = M3_rel_mean(si_fixed,:).';

[C_med,  alpha_med,  yfit_med]  = fit_powerlaw_and_curve(xM, y_med);
[C_mean, alpha_mean, yfit_mean] = fit_powerlaw_and_curve(xM, y_mean);

loglog(xM, y_mean, '-o', 'LineWidth',1.5); hold on;
loglog(xM, y_med,  '--s','LineWidth',1.5);
loglog(xM, yfit_mean, ':', 'LineWidth',1.5);
loglog(xM, yfit_med,  '-.', 'LineWidth',1.5);
grid on;
xlabel('M'); ylabel('Rel. error (M^3)');
title(sprintf('M^3 @ \\sigma = %.3g', snr_factors(si_fixed)*sigma_base));
set(gca,'XScale','log');

legend( ...
  'mean over trials', ...
  'median over trials', ...
  sprintf('fit mean: C=%.3g, \\alpha=%.3f', C_mean, alpha_mean), ...
  sprintf('fit median: C=%.3g, \\alpha=%.3f', C_med,  alpha_med), ...
  'Location','best');

% --- save Figure B ---
%exportgraphics(figB, fullfile(outdir, sprintf('relerr_vs_M_sigmaIdx%d.png', si_fixed)), 'Resolution', 300);
%exportgraphics(figB, fullfile(outdir, sprintf('relerr_vs_M_sigmaIdx%d.pdf', si_fixed)),  'ContentType','vector');
%savefig(figB, fullfile(outdir, sprintf('relerr_vs_M_sigmaIdx%d.fig', si_fixed)));

%% ============ (C) Error vs M, all sigmas on same axes ============
figC = figure('Name','Rel error vs M (all \sigma on same axes)');

% ----- M^2 -----
subplot(1,2,1);
xM = M_list(:);
hold on; grid on; box on;
h_mean_M2 = gobjects(nS,1);

for si = 1:nS
    y_mean = M2_rel_mean(si,:).';
    [C_mean, alpha_mean, yfit_mean] = fit_powerlaw_and_curve(xM, y_mean); 

    % mean line (legend entry)
    h_mean_M2(si) = loglog(xM, y_mean, '-o', ...
        'LineWidth', 1.5, ...
        'DisplayName', sprintf('\\sigma = %.3g\\cdotstd(f)', snr_factors(si)));

    % dashed fit (same color, hidden from legend)
    loglog(xM, yfit_mean, '--', ...
        'LineWidth', 1.2, ...
        'Color', get(h_mean_M2(si),'Color'), ...
        'HandleVisibility','off');

    % annotate alpha near last point
    text(xM(end), y_mean(end), sprintf('  \\alpha=%.3f', alpha_mean), ...
        'Color', get(h_mean_M2(si),'Color'), ...
        'FontSize', 9, 'VerticalAlignment','middle');
end

xlabel('Observations (M)');
ylabel('Rel. error (M^2)');
title('M^2: error vs. M (all \sigma)');
set(gca,'XScale','log','YScale','log');
legend(h_mean_M2, 'Location','southwest', 'NumColumns', max(1,ceil(nS/2)));

% ----- M^3 -----
subplot(1,2,2);
hold on; grid on; box on;
h_mean_M3 = gobjects(nS,1);

for si = 1:nS
    y_mean = M3_rel_mean(si,:).';
    [C_mean, alpha_mean, yfit_mean] = fit_powerlaw_and_curve(xM, y_mean); 

    % mean line (legend entry)
    h_mean_M3(si) = loglog(xM, y_mean, '-o', ...
        'LineWidth', 1.5, ...
        'DisplayName', sprintf('\\sigma = %.3g\\cdotstd(f)', snr_factors(si)));

    % dashed fit (same color, hidden from legend)
    loglog(xM, yfit_mean, '--', ...
        'LineWidth', 1.2, ...
        'Color', get(h_mean_M3(si),'Color'), ...
        'HandleVisibility','off');

    % annotate alpha near last point
    text(xM(end), y_mean(end), sprintf('  \\alpha=%.3f', alpha_mean), ...
        'Color', get(h_mean_M3(si),'Color'), ...
        'FontSize', 9, 'VerticalAlignment','middle');
end

xlabel('Observations (M)');
ylabel('Rel. error (M^3)');
title('M^3: error vs. M (all \sigma)');
set(gca,'XScale','log','YScale','log');
legend(h_mean_M3, 'Location','southwest', 'NumColumns', max(1,ceil(nS/2)));

% --- save Figure C ---
%exportgraphics(figC, fullfile(outdir, 'relerr_vs_M_all_sigma.png'), 'Resolution', 300);
%exportgraphics(figC, fullfile(outdir, 'relerr_vs_M_all_sigma.pdf'),  'ContentType','vector');
%savefig(figC, fullfile(outdir, 'relerr_vs_M_all_sigma.fig'));



%% ================== Helper: relative error ==================
function [C, alpha, yfit] = fit_powerlaw_and_curve(x, y)
% Fits y ~ C * x^{-alpha} using log10 least squares.
% Returns C, alpha, and yfit evaluated on the provided x.
    mask = isfinite(x) & isfinite(y) & (x>0) & (y>0);
    x = x(:); y = y(:);
    x = x(mask); y = y(mask);
    p = polyfit(log10(x), log10(y), 1);  % log10(y) = p1*log10(x) + p2
    alpha = -p(1);
    C = 10^(p(2));
    yfit = C * x.^(-alpha);
end

function v = relerr(x, y)
    den = max(abs(y), 1e-15);
    v = abs(x - y) ./ den;
end

% ================= helpers ==========================
function G = gauss2d(S, mu, sx, sy, rho)
[x,y] = meshgrid(1:S,1:S);
xc = x-mu(1); yc = y-mu(2);
A = 1/(2*pi*sx*sy*sqrt(1-rho^2));
Q = (xc.^2/sx^2 - 2*rho*xc.*yc/(sx*sy) + yc.^2/sy^2) / (2*(1-rho^2));
G = A * exp(-Q);
end

function H = ring_values_one(f, r0, cfg)
% Samples f around the origin on radius r0 at Nphi angles; 'cubic' interp.
S = size(f,1); ctr = (S+1)/2;
Nphi   = cfg.Nphi;
Nphi_os= ternary(~isempty(cfg.Nphi_os) && cfg.Nphi_os > Nphi, cfg.Nphi_os, Nphi);
phi_os = linspace(0, 2*pi*(1-1/Nphi_os), Nphi_os);
Vr     = [r0*cos(phi_os); r0*sin(phi_os)];
Xv     = Vr(1,:).' + ctr;
Yv     = Vr(2,:).' + ctr;
Hos    = interp2(double(f), Xv, Yv, cfg.ringMethod, 0.0);   % [Nphi_os x 1]
if Nphi_os ~= Nphi
    Hos = angular_decimate_os_to_Nphi(Hos(:), Nphi_os, Nphi);  % vector-friendly
end
H = Hos(:).';   % row vector length Nphi
end

function [prep, rings] = build_minimal_pieces(f, cfg, need_r0)
% Small subset used for the grid-FFT baseline (same as earlier demo)
S = size(f,1); ctr = (S+1)/2; fCPU = double(f);
Rdisk = cfg.R; Nphi = cfg.Nphi;

% translations base set
[Xg,Yg] = meshgrid(1:S,1:S);
rr_full = hypot(Xg-ctr, Yg-ctr);
Rbase = (cfg.limit_translations) * min(max(cfg.trans_radius,0), Rdisk - 1e-9) ...
      + (~cfg.limit_translations) * (Rdisk - 1e-9);
base_valid = rr_full <= Rbase;
[y_list, x_list] = find(base_valid);
Xbase = double(x_list); Ybase = double(y_list);

% s(y)
if isfield(cfg,'Ntheta') && ~isempty(cfg.Ntheta), Ntheta = cfg.Ntheta; else, Ntheta = max(180, round(8*pi*Rdisk)); end
bnd_R = Rdisk - cfg.delta_pix;
theta_SE = linspace(0, 2*pi*(1-1/Ntheta), Ntheta);
U0 = [bnd_R*cos(theta_SE); bnd_R*sin(theta_SE)]; U1 = -U0;
U0 = cat(3,U0(1,:),U0(2,:));  U1 = cat(3,U1(1,:),U1(2,:));
v0 = interp2(fCPU, Xbase + U0(:,:,1), Ybase + U0(:,:,2), cfg.bndMethod, 0.0);
v1 = interp2(fCPU, Xbase + U1(:,:,1), Ybase + U1(:,:,2), cfg.bndMethod, 0.0);
s_r = mean(v0 .* v1, 2);
idx_active = find(s_r ~= 0);
sa = s_r(idx_active);

% indices
[m1g,m2g] = ndgrid(0:Nphi-1, 0:Nphi-1);
IDX2D = mod(-(m1g + m2g), Nphi) + 1;
IDX1D = mod(-(0:Nphi-1), Nphi) + 1;
IDX2D_flat = IDX2D(:);

% rings (unitary ring spectra over ACTIVE translations)
need_r0 = unique(need_r0(:).');
rings = cell(1, max(need_r0)+1);
for r0 = need_r0
    rings{r0+1} = ring_spectrum_one(fCPU, Xbase(idx_active), Ybase(idx_active), r0, cfg);
end

prep = struct('sa', sa, 'IDX1D', IDX1D, 'IDX2D_flat', IDX2D_flat);
end

function C_r = ring_spectrum_one(fCPU, Xbase_act, Ybase_act, r0, cfg)
Nphi   = cfg.Nphi;
Nphi_os= ternary(~isempty(cfg.Nphi_os) && cfg.Nphi_os > Nphi, cfg.Nphi_os, Nphi);
phi_os = linspace(0, 2*pi*(1-1/Nphi_os), Nphi_os);
Vr     = [r0*cos(phi_os); r0*sin(phi_os)];
Xv     = Vr(1,:).' + (Xbase_act(:)).';
Yv     = Vr(2,:).' + (Ybase_act(:)).';
Hos    = interp2(fCPU, Xv, Yv, cfg.ringMethod, 0.0);   % [Nphi_os x Ny_act]
if Nphi_os ~= Nphi
    Hos = angular_decimate_os_to_Nphi(Hos, Nphi_os, Nphi);
end
C_r = fft(Hos, [], 1) / sqrt(Nphi);
end

function H = angular_decimate_os_to_Nphi(Hos, Nphi_os, Nphi)
% Vector or matrix input Hos supported.
Ny = size(Hos,2);
C_os = fft(Hos, [], 1) / sqrt(Nphi_os);
if Nphi_os == 2*Nphi && mod(Nphi,2)==0
    half = Nphi/2;
    C_lp = zeros(Nphi, Ny, 'like', C_os);
    C_lp(1,:) = C_os(1,:);
    if half >= 2, C_lp(2:half,:) = C_os(2:half,:); end
    C_lp(half+1,:) = 0.5*(C_os(half+1,:) + C_os(Nphi_os - half + 1,:));
    if half >= 2, C_lp(half+2:end,:) = C_os(Nphi_os - (half-2) : Nphi_os, :); end
    H_unit = real(ifft(C_lp, [], 1) * sqrt(Nphi));
else
    m_os = fftshift((-floor(Nphi_os/2)):(ceil(Nphi_os/2)-1)).';
    m_t  = fftshift((-floor(Nphi/2)) :(ceil(Nphi/2)-1)).';
    Csh  = fftshift(C_os,1);
    C_lp = zeros(Nphi, Ny, 'like', C_os);
    for j = 1:max(1,Ny)
        C_lp(:,j) = interp1(m_os, Csh(:,j), m_t, 'linear', 0.0);
    end
    C_lp = ifftshift(C_lp,1);
    H_unit = real(ifft(C_lp, [], 1) * sqrt(Nphi));
end
H = sqrt(Nphi / Nphi_os) * H_unit;
end

function y = ternary(cond,a,b)
if cond, y=a; else, y=b; end
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
    [Xg,Yg] = meshgrid(1:S,1:S);
    rr=hypot(Xg-ctr,Yg-ctr)/R; ang=atan2(Yg-ctr,Xg-ctr);
    f = exp(-rr.^2).*(0.4 + 0.6*cos(3*ang));
end
f(~inside)=0; f(inside)=mat2gray(f(inside));
end

function m = disk_mask(S, R)
ctr=(S+1)/2; [X,Y]=meshgrid(1:S,1:S); m = hypot(X-ctr,Y-ctr) <= R;
end
