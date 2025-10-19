function [theta_est, info] = sync_ring_rotations_from_M2(rings_rot, M2, Nphi, opts)
%SYNC_RING_ROTATIONS_FROM_M2  Estimate per-ring rotations from M² only.
%   [theta_est, info] = SYNC_RING_ROTATIONS_FROM_M2(rings_rot, M2, Nphi, opts)
%   performs inter-ring rotation synchronization using only second-order
%   autocorrelations (M²). It pools information across multiple harmonics
%   k=±1..±Kmax via a k-root trick and builds a weighted synchronization
%   graph, followed by a spectral initialization.
%
%   INPUTS
%     rings_rot : [R x Nphi] real. Each row is a ring signal (possibly rotated).
%     M2        : {R x R} cell. M2{i,j} is [1 x Nphi] and equals the φ-domain
%                 non-unitary M² between rings i and j, aligned to a common grid.
%                 (Used only through its FFT.)
%     Nphi      : Positive integer. Angular sampling size.
%     opts      : struct with fields:
%                   .Kmax            (required, >0) max |k| harmonic to use (excl. k=0)
%                   .min_edge_weight (default 1e-6) threshold to keep graph edges
%                   .verbose         (default true) print diagnostics
%
%   OUTPUTS
%     theta_est : [R x 1] estimated ring rotations (radians), gauge-fixed so that
%                 theta_est(1) = 0.
%     info      : struct with debug fields:
%                   .edges [m x 2], .vals [m x 1] (edge offsets),
%                   .wts [m x 1] (edge weights), .Kmax, .Nphi
%
%   METHOD (high level)
%     1) Compute Ahat = FFT(rings_rot, φ)/Nphi to obtain complex amplitudes.
%     2) For each pair (i,j) and harmonic k, form q_k ≈ e^{i k (θ_i - θ_j)}
%        via q_k = (Ahat_i[k] * conj(Ahat_j[k])) / C_ij[k] where C_ij = FFT(M2_ij).
%     3) Pool over k using the k-root map: exp(i * angle(q_k)/k), then take a
%        circular mean with SNR-like weights to estimate Δθ_ij and an edge weight.
%     4) Build a weighted Hermitian matrix and extract the leading eigenvector
%        (spectral sync). Fix gauge so that θ_1 = 0.
%
%   NOTES
%     - Uses MATLAB's non-unitary FFT conventions.
%     - Requires that M2_ij and ring spectra are on the same angular grid.
%     - Robustness: edges with weak/unstable pooled evidence are dropped via
%       min_edge_weight; an additional residual-based pruning is applied.
%
%   SEE ALSO: FFT, IFFT

if nargin < 4, opts = struct(); end
assert(isfield(opts,'Kmax') && opts.Kmax>0, 'opts.Kmax must be provided and > 0');
Kmax = opts.Kmax;
min_edge_weight = get_opt(opts,'min_edge_weight',1e-6);
verbose         = get_opt(opts,'verbose',true);

[R, Nphi_check] = size(rings_rot);
assert(Nphi_check==Nphi, 'rings_rot must be R x Nphi');

% Spectra (non-unitary)
Ahat = fft(rings_rot, [], 2)/Nphi;   % R x Nphi

% Build C_ij[k] = FFT{M2_ij(φ)}(k)
C = cell(R,R);
for i=1:R
  for j=i:R
    Cij = fft(M2{i,j}, [], 2);       % 1 x Nphi
    C{i,j} = Cij;
    C{j,i} = conj(Cij);
  end
end

% Usable harmonics (exclude DC)
k_all  = ifftshift(-floor(Nphi/2):ceil(Nphi/2)-1);
k_use  = k_all(abs(k_all) <= Kmax & k_all ~= 0);
idx_use = mod(k_use, Nphi) + 1;

% Build edges
edges = []; vals = []; wts = []; resids = [];
for i=1:R
  for j=i+1:R
    num = Ahat(i,idx_use) .* conj(Ahat(j,idx_use));
    den = C{i,j}(idx_use);
    mask_good = (abs(den) > 1e-12) & (abs(num) > 1e-12);
    if ~any(mask_good), continue; end
    qk = num(mask_good) ./ den(mask_good);    % ≈ e^{i k (θ_i - θ_j)}
    kk = k_use(mask_good);
    idxk = idx_use(mask_good);

    % Weights (proxy for SNR)
    wk = abs(den(mask_good)) .* abs(Ahat(i,idxk)) .* abs(Ahat(j,idxk));

    % Multi-k pooling via k-root map then circular mean
    angQ = angle(qk);
    zk = exp(1i * (angQ ./ kk));             % pull phases together
    zbar = circular_mean(zk, wk);
    dtheta = angle(zbar);
    Wedge  = abs(zbar) * sum(wk);

    % Residual consistency across k
    res = sum(wk .* (1 - cos(angQ - kk*dtheta))) / max(sum(wk),1e-12);

    if Wedge >= min_edge_weight
        edges(end+1,:) = [i,j]; 
        vals(end+1,1)  = dtheta; 
        wts(end+1,1)   = Wedge;  
        resids(end+1,1)= res;    
    end
  end
end

if verbose && ~isempty(wts)
  fprintf('[EDGES] kept %d/%d edges | |W|: min=%.3e med=%.3e max=%.3e | resid(med)=%.3e\n', ...
    size(edges,1), R*(R-1)/2, min(wts), median(wts), max(wts), median(resids));
end

% Optional: prune worst residuals (keep strong or consistent)
if ~isempty(resids)
    q95 = quantile(resids, 0.95);
    keep = (resids <= q95) | (wts >= median(wts));
    edges = edges(keep,:); vals = vals(keep); wts = wts(keep);
    if verbose
      fprintf('[PRUNE] kept %d edges after residual filter\n', size(edges,1));
    end
end

% Spectral synchronization
theta_est = spectral_sync_from_edges(R, edges, vals, wts);
theta_est = angle(exp(1i*(theta_est - theta_est(1)))); % gauge

% Output
info = struct('edges',edges,'vals',vals,'wts',wts,'Kmax',Kmax,'Nphi',Nphi);
end

% ===== sync helpers =====
function z = circular_mean(zs, w)
%CIRCULAR_MEAN  Weighted circular mean of complex phases on the unit circle.
%   z = CIRCULAR_MEAN(zs, w) returns the complex mean z = sum(w.*zs)/sum(w),
%   where zs lie on the unit circle. If w is omitted, equal weights are used.
%   Returns z=0 if the (weighted) sum cancels exactly.
if nargin<2 || isempty(w), w = ones(size(zs)); end
z = sum(w(:).*zs(:)) / max(sum(w),1e-12);
if z==0, z = 0; end
end

function val = get_opt(opts, field, default_val)
%GET_OPT  Utility: opts.field with default fallback.
%   val = GET_OPT(opts, field, default_val) returns opts.(field) if it exists,
%   otherwise returns default_val.
if isfield(opts, field), val = opts.(field); else, val = default_val; end
end

function theta = spectral_sync_from_edges(R, edges, dtheta, w)
%SPECTRAL_SYNC_FROM_EDGES  Spectral estimator on a weighted rotation graph.
%   theta = SPECTRAL_SYNC_FROM_EDGES(R, edges, dtheta, w) builds a Hermitian
%   matrix H from edge offsets dtheta (in radians) and weights w, and returns
%   per-node phases theta from the leading eigenvector of (H+H')/2. The global
%   gauge is fixed so that theta(1)=0 (wrapped to (-π,π]).
%
%   INPUTS
%     R      : number of nodes (rings).
%     edges  : [m x 2] integer pairs (i,j), i<j.
%     dtheta : [m x 1] estimated relative rotations (θ_i - θ_j).
%     w      : [m x 1] nonnegative edge weights.
%
%   OUTPUT
%     theta  : [R x 1] node phases (radians), gauge-fixed.
H = zeros(R,R);
m = size(edges,1);
for e=1:m
    i = edges(e,1); j = edges(e,2);
    Hij = w(e) * exp(1i*dtheta(e));
    H(i,j) = H(i,j) + Hij;
    H(j,i) = H(j,i) + conj(Hij);
end
% leading eigenvector
if R <= 400
    [V,D] = eig((H+H')/2);
    [~,ix] = max(real(diag(D)));
    u = V(:,ix);
else
    try
        u = eigs((H+H')/2, 1, 'la');
    catch
        [V,D] = eig((H+H')/2);
        [~,ix] = max(real(diag(D)));
        u = V(:,ix);
    end
end
theta = angle(u);
theta = angle(exp(1i*(theta - theta(1))));
end
