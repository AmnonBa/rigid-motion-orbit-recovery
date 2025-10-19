function [M3_red, M2_red, meta, M2_dense, M3s_red_matched, M2_num, M3_num, D_scalar] = ...
    se2_to_so2_M2M3_uniform_efficient(f, cfg, mk0_keep, mk1_keep, M3s_target)
%SE2_TO_SO2_M2M3_UNIFORM_EFFICIENT  SE(2)→SO(2) reduction of M² (all pairs) and M³ (subset).
%   [M3_red, M2_red, meta, M2_dense, M3s_red_matched, M2_num, M3_num, D_scalar] = ...
%      SE2_TO_SO2_M2M3_UNIFORM_EFFICIENT(f, cfg, mk0_keep, mk1_keep, M3s_target)
%   computes second- and third-order angular statistics on rings (SO(2)) from a
%   2-D image f under SE(2) rigid motions, using a GPU-aware, chunked pipeline.
%
%   This version uses a SINGLE global denominator D_scalar = sum_y s(y) for both
%   M² and M³, and builds numerators WITHOUT visibility masks (i.e., weights are
%   purely s(y); no vpair/vtri). The M² numerator uses s(y) directly; the M³
%   numerator multiplies the triple product by s(y) ONCE.
%
%   INPUTS
%     f           : [S x S] real image on a square Cartesian grid.
%     cfg         : struct with required fields:
%                      .R                (disk radius in pixels)
%                      .pad              (half canvas size; image assumed centered)
%                      .Nphi             (angular samples per ring)
%                    and optional fields (defaults shown):
%                      .useGPU           = parallel.gpu.GPUDevice.isAvailable
%                      .verbose          = true
%                      .phi_chunk        = 60        % angular chunk size (oversampled grid)
%                      .r_chunk          = 8         % how many rings per outer loop block
%                      .pair_chunk       = 256       % pair-batch for M² accumulation
%                      .delta_pix        = 0.6       % boundary radius offset
%                      .margin_ring      = 0.5       % margin for last ring
%                      .Nphi_os          = []        % if >Nphi, oversample then decimate
%                      .ringMethod       = 'cubic'   % interp2 for ring sampling
%                      .bndMethod        = 'cubic'   % interp2 for boundary factor
%                      .allow_partial_last_ring = false
%                      .limit_translations = false
%                      .trans_radius     = cfg.pad   % if throttling translations
%                      .prune_by_sabs    = false     % prune translations by |s(y)|
%                      .sabs_quantile    = 0.0
%                      .sabs_min         = 0.0
%                      .max_translations = []        % cap on translations after pruning
%                      .triplets_mode    = 'none'    % 'none'|'explicit'|'window'|'window3'
%                                                   % |'adjacent3'|'all'|'random'
%                      .triplet_offsets  = []        % for 'window3'
%                      .triplets_list    = []        % for 'explicit' (0-based, rows)
%                      .max_triplets     = []        % cap on M³ triplets
%                      .window_hop       = 1         % for 'window'
%                      .triplets_seed    = []        % for 'random'
%                      .match_direct_norm = true     % export DIRECT-shaped normalization
%                      .Ntheta           = []        % #samples for boundary integral s(y)
%                      .y_chunk          = 256       % CPU chunk size for M³ accumulation
%     mk0_keep    : vector of kept 0-based harmonics for B-spectrum axis 1 (used for export).
%     mk1_keep    : vector of kept 0-based harmonics for B-spectrum axis 2 (used for export).
%     M3s_target  : (optional) struct to force DIRECT-style sampling export:
%                     .r1,.r2,.r3 (1-based radii), .t1,.t2 (indices in mk0_keep),
%                     .mode fields, etc. If provided, exports are matched to this list.
%
%   OUTPUTS
%     M3_red  : struct with fields
%                 .Nphi      = Nphi
%                 .triplets  = [Ntrip x 3] 0-based sorted (r1,r2,r3)
%                 .data      = [Nphi x Nphi x Ntrip] Δ-domain M³ numerators / D_scalar
%                 .denom     = [Ntrip x 1] all equal to D_scalar (scalar denom)
%     M2_red  : struct with fields
%                 .Nphi      = Nphi
%                 .pairs     = [Npairs x 2] 0-based ring pairs (r1,r2), r1≤r2
%                 .data      = [Nphi x Npairs] Δ-domain M² numerators / D_scalar
%                 .denom     = [Npairs x 1] all equal to D_scalar
%     meta    : struct with timing, debug info, and the cfg used.
%     M2_dense: [nr x nr x Nphi] DIRECT-shaped normalized M² (Δ-domain), using
%               Δ↔−Δ symmetry for (i,j)↔(j,i). If cfg.match_direct_norm is true,
%               uses the non-unitary ifft normalization factor for export.
%     M3s_red_matched : struct of DIRECT-style sampled bispectrum entries with fields
%               r1,r2,r3 (1-based radii), t1,t2,t3 (indices in kept harmonic sets),
%               beta (phase), W3mag (magnitude), mode (string).
%     M2_num  : [nr x nr x Nphi] raw Δ-domain M² numerators (GPU array if enabled).
%     M3_num  : [Nphi x Nphi x Ntrip] raw Δ-domain M³ numerators (CPU).
%     D_scalar: scalar denominator = sum_y s(y).
%
%   DEFINITIONS (key ideas)
%     • Rings: for radii r=0..Rdisk, sample angular profiles with Nphi points.
%     • Boundary factor: s(y) = mean_θ f(y+U0(θ)) f(y+U1(θ)), with U1 = −U0 at
%       radius bnd_R = R − delta_pix; it summarizes boundary “visibility”.
%     • Non-unitary vs unitary: internal spectral work is mostly unitary in the
%       ring sampling (fft/√Nphi), but DIRECT-shaped exports use non-unitary Δ-domain.
%     • Denominator: D_scalar = ∑_y s(y) (single scalar used for both M² and M³).
%
%   PERFORMANCE / NUMERICAL NOTES
%     • GPU-aware: spectra per ring are kept on GPU when available; M³ accumulation
%       runs on CPU with chunking (y_chunk).
%     • Oversampling: if Nphi_os>Nphi, angular sampling is oversampled then decimated
%       in the Fourier domain with energy-aware downsampling.
%     • Symmetry: M² dense tensor uses Δ↔−Δ indexing symmetry when writing [i,j] and [j,i].
%     • Triplets: chosen per cfg.triplets_mode or matched to M3s_target if provided.
%
%   SEE ALSO: FFT, IFFT, IFFT2, INTERP2
%
%   Author: <your name>, <affiliation> (<year>)
%
% -------------------------------------------------------------------------
% SE2_TO_SO2_M2M3_UNIFORM_EFFICIENT_V5 (Flattened numerators, single scalar denom)
% -------------------------------------------------------------------------
% SE(2) → SO(2) reduction of M² (all pairs) and M³ (subset of triplets).
% This version:
%   • Uses a SINGLE global denominator D_scalar = sum_y s(y) for both M², M³
%   • Numerators DO NOT use visibility: no vpair / vtri (weights purely s(y))
%   • M² numerator uses s(y) directly; M³ multiplies the triple product by s(y) ONCE
%   • Returns raw numerators (M2_num, M3_num) and D_scalar
%   • Emits DIRECT-shaped M2_dense and a DIRECT-style M3 sampling list
%
% s(y) = mean_θ f(y+U0(θ)) f(y+U1(θ)), with U1 = -U0.
% -------------------------------------------------------------------------

% -------- required --------
must = {'R','pad','Nphi'};
for k = 1:numel(must), assert(isfield(cfg,must{k}), 'cfg.%s is required', must{k}); end

% -------- defaults --------
if ~isfield(cfg,'useGPU'),  cfg.useGPU = parallel.gpu.GPUDevice.isAvailable; end
if ~isfield(cfg,'verbose'), cfg.verbose = true; end
if ~isfield(cfg,'phi_chunk') || isempty(cfg.phi_chunk), cfg.phi_chunk = 60; end
if ~isfield(cfg,'r_chunk')  || isempty(cfg.r_chunk),  cfg.r_chunk  = 8; end
if ~isfield(cfg,'pair_chunk')  || isempty(cfg.pair_chunk),  cfg.pair_chunk  = 256; end
if ~isfield(cfg,'delta_pix') || isempty(cfg.delta_pix), cfg.delta_pix = 0.6; end
if ~isfield(cfg,'margin_ring') || isempty(cfg.margin_ring), cfg.margin_ring = 0.5; end
if ~isfield(cfg,'Nphi_os') || isempty(cfg.Nphi_os), cfg.Nphi_os = []; end
if ~isfield(cfg,'ringMethod') || isempty(cfg.ringMethod), cfg.ringMethod = 'cubic'; end
if ~isfield(cfg,'bndMethod')  || isempty(cfg.bndMethod),  cfg.bndMethod  = 'cubic'; end
if ~isfield(cfg,'allow_partial_last_ring'), cfg.allow_partial_last_ring = false; end

% translation throttling
if ~isfield(cfg,'limit_translations'), cfg.limit_translations = false; end
if ~isfield(cfg,'trans_radius') || isempty(cfg.trans_radius), cfg.trans_radius = cfg.pad; end
if ~isfield(cfg,'prune_by_sabs'), cfg.prune_by_sabs = false; end
if ~isfield(cfg,'sabs_quantile') || isempty(cfg.sabs_quantile), cfg.sabs_quantile = 0.0; end
if ~isfield(cfg,'sabs_min') || isempty(cfg.sabs_min), cfg.sabs_min = 0.0; end
if ~isfield(cfg,'max_translations') || isempty(cfg.max_translations), cfg.max_translations = []; end

% triplet selection (used only if M3s_target not provided)
if ~isfield(cfg,'triplets_mode') || isempty(cfg.triplets_mode), cfg.triplets_mode = 'none'; end
if ~isfield(cfg,'triplet_offsets'), cfg.triplet_offsets = []; end
if ~isfield(cfg,'triplets_list'),  cfg.triplets_list  = []; end
if ~isfield(cfg,'max_triplets') || isempty(cfg.max_triplets), cfg.max_triplets = []; end
if ~isfield(cfg,'window_hop') || isempty(cfg.window_hop), cfg.window_hop = 1; end
if ~isfield(cfg,'triplets_seed'), cfg.triplets_seed = []; end
if ~isfield(cfg,'match_direct_norm') || isempty(cfg.match_direct_norm)
    cfg.match_direct_norm = true;  % export in DIRECT normalization for M2 and Bspec sampling
end

% -------- setup --------
useGPU = logical(cfg.useGPU) && parallel.gpu.GPUDevice.isAvailable;
if useGPU, toGPU = @(x) gpuArray(x); toCPU = @(x) gather(x);
else,      toGPU = @(x) x;          toCPU = @(x) x;      end

t_all = tic;
S = size(f,1);
ctr = (S+1)/2;
Rdisk = cfg.R;
Nphi   = cfg.Nphi;

if ~isempty(cfg.Nphi_os) && cfg.Nphi_os > Nphi
    Nphi_os = cfg.Nphi_os;
else
    Nphi_os = Nphi;
end

if isfield(cfg,'Ntheta') && ~isempty(cfg.Ntheta) && cfg.Ntheta > 0
    Ntheta = cfg.Ntheta;
else
    Ntheta = max(180, round(8*pi*Rdisk));
end

% max radius used (inclusive, 0-based)
if isfield(cfg,'max_r') && ~isempty(cfg.max_r)
    max_r = min(cfg.max_r, Rdisk - (~cfg.allow_partial_last_ring));
else
    max_r = Rdisk - (~cfg.allow_partial_last_ring);
end
nr = max_r + 1;

phi_os   = linspace(0, 2*pi*(1-1/Nphi_os), Nphi_os);
theta_SE = linspace(0, 2*pi*(1-1/Ntheta ), Ntheta );
bnd_R    = Rdisk - cfg.delta_pix;

% -------- list ALL M² pairs; M³ triplets per cfg or TARGET --------
pairs    = all_upper_pairs_0based(max_r);   % ALL pairs for M²

force_target = (nargin >= 5) && ~isempty(M3s_target) && isstruct(M3s_target) ...
               && isfield(M3s_target,'r1') && isfield(M3s_target,'r2') && isfield(M3s_target,'r3');

if force_target
    T = sort([M3s_target.r1(:)-1, M3s_target.r2(:)-1, M3s_target.r3(:)-1], 2);
    in = all(T>=0,2) & all(T<=max_r,2);
    triplets = unique(T(in,:), 'rows');
    triplet_mode_used = 'direct_target';
else
    triplets = build_triplets(cfg, max_r, nr);
    triplet_mode_used = cfg.triplets_mode;
end

Npairs   = size(pairs,1);
Ntrip    = size(triplets,1);
if cfg.verbose
    fprintf('[RED] planning: nr=%d rings (0..%d) | pairs=%d | triplets=%d (mode=%s)\n', ...
        nr, max_r, Npairs, Ntrip, triplet_mode_used);
end

% -------- precompute ring offsets --------
v_ring_all = zeros(2, Nphi_os, nr);
for ir = 1:nr
    r = ir-1;
    v_ring_all(:,:,ir) = [r*cos(phi_os); r*sin(phi_os)];
end

% -------- translations base set --------
[Xg,Yg] = meshgrid(1:S, 1:S);
rr_full = hypot(Xg-ctr, Yg-ctr);

if cfg.limit_translations
    Rbase = min(max(cfg.trans_radius,0), Rdisk - 1e-9);
else
    Rbase = Rdisk - 1e-9;
end
base_valid = rr_full <= Rbase;
[y_list, x_list] = find(base_valid);
Ny_all = numel(y_list);  assert(Ny_all>0, 'Translation base set is empty.');

Xbase = double(x_list);
Ybase = double(y_list);
rr_pts= hypot(Xbase-ctr, Ybase-ctr);

% -------- boundary factor s(y) --------
U0 = [bnd_R*cos(theta_SE); bnd_R*sin(theta_SE)];   U1 = -U0;
U0 = cat(3,U0(1,:),U0(2,:));  U1 = cat(3,U1(1,:),U1(2,:));

fCPU = double(f);
v0 = interp2(fCPU, Xbase + U0(:,:,1), Ybase + U0(:,:,2), cfg.bndMethod, 0.0);
v1 = interp2(fCPU, Xbase + U1(:,:,1), Ybase + U1(:,:,2), cfg.bndMethod, 0.0);
s_r   = mean(v0 .* v1, 2);   % s(y)
s_abs = abs(s_r);

% prune/cap translations
keep = true(Ny_all,1);
if cfg.prune_by_sabs
    thr = cfg.sabs_min;
    if cfg.sabs_quantile > 0 && cfg.sabs_quantile < 1, thr = max(thr, quantile(s_abs, cfg.sabs_quantile)); end
    keep = (s_abs >= thr);
end
if ~isempty(cfg.max_translations) && cfg.max_translations < sum(keep)
    [~,ord] = maxk(s_abs, cfg.max_translations);
    kmask = false(Ny_all,1); kmask(ord) = true;
    keep = keep & kmask;
end

Xbase = Xbase(keep);  
Ybase = Ybase(keep);  
rr_pts = rr_pts(keep);
s_r   = s_r(keep);    
Ny    = numel(Xbase); 
assert(Ny>0, 'All translations pruned.');

% weights
w_base  = s_r(:);            % for denominator: sum_y s(y)
s_r_gpu = toGPU(s_r(:));     % for numerators

% -------- index maps --------
[m1g,m2g] = ndgrid(0:Nphi-1, 0:Nphi-1);
IDX2D = toGPU(mod(-(m1g + m2g), Nphi) + 1);          % index for m3 = -(m1+m2)
IDX1D = toGPU(mod(-(0:Nphi-1), Nphi) + 1);           % index for -m
IDX1D_cpu = mod(-(0:Nphi-1), Nphi) + 1;              % CPU copy (for symmetry writes)

% -------- sample ALL rings (0..max_r) once --------
C_all = cell(nr,1);        % each: [Nphi x Ny]
valid_frac = cell(nr,1);   % kept for completeness (unused in numerators here)

decimator_used = (Nphi_os ~= Nphi);
decim_energy_before = 0; decim_energy_after = 0;

phi_chunk = max(1, min(cfg.phi_chunk, Nphi_os));
r_chunk   = max(1, min(cfg.r_chunk,  nr));

for rr0 = 1:r_chunk:nr
    rr1 = min(nr, rr0 + r_chunk - 1);
    for ir = rr0:rr1
        r = ir-1;
        if cfg.allow_partial_last_ring
            idx_trans = (1:Ny).';
        else
            idx_trans = find(rr_pts <= (Rdisk - cfg.margin_ring - r));
        end
        Ny_r = numel(idx_trans);

        H_os = zeros(Nphi_os, Ny_r);
        if cfg.allow_partial_last_ring, valid_cnt = zeros(Ny_r,1); end

        Vr_all = v_ring_all(:,:,ir);
        for k = 1:phi_chunk:Nphi_os
            kk = k:min(Nphi_os, k+phi_chunk-1);
            Vr = Vr_all(:,kk);
            Xv = Vr(1,:).' + (Xbase(idx_trans)).';
            Yv = Vr(2,:).' + (Ybase(idx_trans)).';
            H_os(kk,:) = interp2(fCPU, Xv, Yv, cfg.ringMethod, 0.0);
            if cfg.allow_partial_last_ring
                Ssz = size(fCPU,1);
                M = (Xv >= 1) & (Xv <= Ssz) & (Yv >= 1) & (Yv <= Ssz) ...
                    & ((Xv-ctr).^2 + (Yv-ctr).^2 <= (Rdisk+1e-9)^2);
                valid_cnt = valid_cnt + sum(M,1).';
            end
        end

        if decimator_used
            decim_energy_before = decim_energy_before + sum(H_os(:).^2);
            H = angular_decimate_os_to_Nphi_gpuaware(H_os, Nphi_os, Nphi, useGPU);
            decim_energy_after  = decim_energy_after  + sum(gather(H(:)).^2);
        else
            H = toGPU(H_os);
        end

        C_loc = toGPU(zeros(Nphi, Ny, 'double'));
        vfrac = toGPU(zeros(Ny,1,'double'));

        C_loc(:, idx_trans) = fft(H, [], 1) / sqrt(Nphi);  % UNITARY spectra
        if cfg.allow_partial_last_ring
            vfrac(idx_trans) = toGPU(max(valid_cnt / Nphi_os, 0));
        else
            vfrac(idx_trans) = 1;
        end

        C_all{ir} = C_loc;
        valid_frac{ir} = vfrac;
    end
end

% -------- alloc accumulators (M2_num is DENSE: [nr x nr x Nphi]) --------
M2_num = toGPU(zeros(nr, nr, Nphi, 'double'));
if Ntrip > 0
    M3_num    = toGPU(zeros(Nphi, Nphi, Ntrip));   % Δ-domain numerators (real after ifft2)
    Bspec_num = toGPU(zeros(Nphi, Nphi, Ntrip));   % COMPLEX bispectrum numerators
else
    M3_num    = toGPU(zeros(Nphi, Nphi, 0));
    Bspec_num = toGPU(zeros(Nphi, Nphi, 0));
end

% -------- M² fast over pair chunks (NO visibility in numerators) --------
Nphi_chk = size(C_all{1},1); 
Ny_chk   = size(C_all{1},2); 

% Numerator weights for M²: use s(y) directly
w_eff_full = s_r_gpu;                        % Ny x 1

% Build C_neg once (rows mapped by -m)
C_neg = cell(size(C_all));
for r = 1:numel(C_all)
    C_neg{r} = C_all{r}(IDX1D, :);           % Nphi x Ny
end

% Optional: prune zero-weight samples globally for efficiency
mask_cols = (w_base ~= 0);
if any(~mask_cols)
    w_base_eff = w_base(mask_cols);          % for scalar denom
    w_eff = w_eff_full(mask_cols);           % for numerators
else
    w_base_eff = w_base;
    w_eff = w_eff_full;
end
Ny_eff = numel(w_eff);

% SINGLE global denominator
if any(~mask_cols)
    D_scalar = sum(w_base_eff);              % sum_y s(y)
else
    D_scalar = sum(w_base);                  % sum_y s(y)
end
D_scalar = double(gather(D_scalar));

% Preallocate batch workspaces once (max chunk size)
pair_chunk = max(1, min(cfg.pair_chunk, Npairs));
Kb_max = pair_chunk;
X     = zeros(Nphi, Ny_eff, Kb_max, 'like', C_all{1});  % Nphi x Ny_eff x Kb
Alpha = zeros(Ny_eff, 1, Kb_max, 'like', w_eff);        % Ny_eff x 1 x Kb

% Main loop over pair chunks
for k0 = 1:pair_chunk:Npairs
    k1 = min(Npairs, k0 + pair_chunk - 1);
    kk = k0:k1;  Kb = numel(kk);

    % ---- build the batch (Hadamards, no vpair) ----
    for j = 1:Kb
        k   = kk(j);
        r1  = pairs(k,1);
        r2  = pairs(k,2);

        C1   = C_all{r1+1};
        C2n  = C_neg{r2+1};

        % NO visibility: vpair = 1
        Alpha(:,1,j) = w_eff;                 % Ny_eff x 1
        if any(~mask_cols)
            X(:,:,j)     = C1(:,mask_cols) .* C2n(:,mask_cols);
        else
            X(:,:,j)     = C1 .* C2n;
        end
    end

    % ---- one batched weighted sum over samples (y) for all pairs ----
    Sspec_batch = pagemtimes(X(:,:,1:Kb), Alpha(:,:,1:Kb));   % Nphi x 1 x Kb
    Sspec_batch = reshape(Sspec_batch, Nphi, Kb);             % Nphi x Kb

    % ---- vectorized IFFT across the Kb columns ----
    M2_cols = real(ifft(Sspec_batch, [], 1)) * sqrt(Nphi);  % Nphi x Kb

    % ---- write each column into dense [nr x nr x Nphi] with Δ↔−Δ symmetry ----
    for j = 1:Kb
        kpair = kk(j);
        i = pairs(kpair,1) + 1;   % 1-based r1
        r = pairs(kpair,2) + 1;   % 1-based r2
        col = M2_cols(:,j);       % Nphi x 1 (Δ-domain)
        M2_num(i,r,:) = reshape(col, 1,1,[]);
        M2_num(r,i,:) = reshape(col(IDX1D_cpu), 1,1,[]);
    end
end

% ======= M³ (CPU-only, chunked & reuse masked spectra) =======
IDX2D_flat = IDX2D(:);  % Nphi^2 x 1

if Ntrip > 0
    % 1) Mask once
    idx_active = find(s_r ~= 0);
    Ny_active  = numel(idx_active);
    if Ny_active == 0
        M3_num(:)    = 0;
        Bspec_num(:) = 0;
    else
        sa = s_r(idx_active);                    % Ny_active x 1  (CPU)
        % Optional: choose a chunk size (tune if needed)
        if isfield(cfg,'y_chunk') && ~isempty(cfg.y_chunk)
            y_chunk = max(1, cfg.y_chunk);
        else
            y_chunk = 256;                        % good default for CPU
        end

        % 2) Precompute masked spectra for all radii once
        Cmask = cell(nr,1);                      % each: [Nphi x Ny_active]
        for r = 1:nr
            Cmask{r} = C_all{r}(:, idx_active);
        end

        % 3) Triplet accumulation (chunk over active translations)
        parfor t = 1:Ntrip
            r1 = triplets(t,1)+1; r2 = triplets(t,2)+1; r3 = triplets(t,3)+1;
            C1 = Cmask{r1};                      % [Nphi x Ny_active]
            C2 = Cmask{r2};
            C3 = Cmask{r3};

            Bsum = zeros(Nphi, Nphi);            % complex accumulator

            for y0 = 1:y_chunk:Ny_active
                y1  = min(Ny_active, y0 + y_chunk - 1);
                yc  = y0:y1;                      % indices in masked set
                K   = numel(yc);

                % Outer products for K translations in one go
                A = reshape(C1(:,yc), Nphi, 1, K);
                B = reshape(C2(:,yc), 1, Nphi, K);
                OP = pagemtimes(A, B);           % [Nphi x Nphi x K]

                % c3(-m1-m2, y) pages
                C3neg_pages = reshape(C3(IDX2D_flat, yc), Nphi, Nphi, K);

                % Apply s(y) once and accumulate over the 3rd dim
                S = reshape(sa(yc), 1, 1, K);    % broadcast to pages
                Bsum = Bsum + sum(OP .* C3neg_pages .* S, 3);
            end

            % Save numerators
            Bspec_num(:,:,t) = Bsum;                         % complex bispectrum numerator
            M3_num(:,:,t)    = real(ifft2(Bsum)) * Nphi;     % Δ-domain numerator (real)
        end
    end
end

% -------- normalize & pack "ratio" outputs ----------------------
M2_red = struct('Nphi', Nphi, 'pairs', pairs, 'data', [], 'denom', []);
M3_red = struct('Nphi', Nphi, 'triplets', triplets, 'data', [], 'denom', []);

% M² with scalar denom — extract columns from dense tensor
M2_red.data  = zeros(Nphi, Npairs, 'double');
M2_red.denom = repmat(D_scalar, Npairs, 1);
for k = 1:Npairs
    i = pairs(k,1) + 1;
    j = pairs(k,2) + 1;
    col_num = toCPU(reshape(M2_num(i,j,:), Nphi, 1));   % numerator slice
    if abs(D_scalar) > 1e-30
        M2_red.data(:,k) = col_num / D_scalar;
    else
        M2_red.data(:,k) = 0;
    end
end

% M³ with scalar denom
if Ntrip > 0
    M3_red.data  = toCPU(zeros(Nphi, Nphi, Ntrip));
    M3_red.denom = repmat(D_scalar, Ntrip, 1);
    for t = 1:Ntrip
        if abs(D_scalar) > 1e-30
            M3_red.data(:,:,t) = toCPU(M3_num(:,:,t) / D_scalar);
        else
            M3_red.data(:,:,t) = 0;
        end
    end
else
    M3_red.data  = zeros(Nphi,Nphi,0);
    M3_red.denom = zeros(0,1);
end

% -------- also emit DIRECT-shaped outputs ----------------------
% (1) Dense M2 [nr x nr x Nphi] using Δ↔−Δ symmetry (normalized)
M2_dense = zeros(nr, nr, Nphi);
scale_M2_export = ternary(cfg.match_direct_norm, Nphi, 1.0);  % DIRECT uses non-unitary ifft norm

for k = 1:Npairs
    i = M2_red.pairs(k,1) + 1;   % 1-based
    j = M2_red.pairs(k,2) + 1;
    col = M2_red.data(:,k) * scale_M2_export;
    M2_dense(i,j,:) = col(:);
    M2_dense(j,i,:) = col(IDX1D_cpu);
end

% (2) Sampled M³ in DIRECT style — from COMPLEX Bspec numerators
M3s_red_matched = struct('r1',[],'r2',[],'r3',[], ...
                         't1',[],'t2',[],'t3',[], ...
                         'beta',[],'W3mag',[],'mode','reduction');

% Build a LUT from (r1,r2,r3) (1-based) → triplet index t
TripLUT = containers.Map('KeyType','char','ValueType','int32'); 
for t = 1:Ntrip
    r0 = triplets(t,:);  % 0-based sorted
    TripLUT(sprintf('%d_%d_%d', r0(1)+1, r0(2)+1, r0(3)+1)) = t;
end

if Ntrip > 0 && ~isempty(mk0_keep) && ~isempty(mk1_keep)
    Nm = numel(mk0_keep);
    mapH = containers.Map(num2cell(mk0_keep), num2cell(1:Nm)); % 0-based m → kept-index
    take_these_triplets = true(1, Ntrip);

    if force_target
        take_these_triplets(:) = false;
        T0 = sort([M3s_target.r1(:)-1, M3s_target.r2(:)-1, M3s_target.r3(:)-1], 2);
        in = all(T0>=0,2) & all(T0<=max_r,2);
        T0 = unique(T0(in,:), 'rows');
        Kset = containers.Map('KeyType','char','ValueType','logical');
        for tt=1:size(T0,1)
            Kset(sprintf('%d_%d_%d',T0(tt,1),T0(tt,2),T0(tt,3))) = true;
        end
        for t=1:Ntrip
            key = sprintf('%d_%d_%d', triplets(t,1), triplets(t,2), triplets(t,3));
            if isKey(Kset,key), take_these_triplets(t)=true; end
        end
        M3s_red_matched.mode = 'reduction-matched';
    end

    for t = find(take_these_triplets)
        % COMPLEX bispectrum per triplet with scalar denom
        if abs(D_scalar) > 1e-30
            Bspec = toCPU(Bspec_num(:,:,t)) / D_scalar;
        else
            Bspec = zeros(Nphi,Nphi);
        end

        if force_target
            r1o = triplets(t,1)+1; r2o = triplets(t,2)+1; r3o = triplets(t,3)+1;
            idx_dir = find( (M3s_target.r1==r1o) & (M3s_target.r2==r2o) & (M3s_target.r3==r3o) );
            for q = idx_dir(:).'
                mi0 = mk0_keep( M3s_target.t1(q) );  % 0-based m1
                mj0 = mk0_keep( M3s_target.t2(q) );  % 0-based m2
                if ~isKey(mapH, mi0) || ~isKey(mapH, mj0), continue; end
                Bij = Bspec(mi0+1, mj0+1);
                M3s_red_matched.r1(end+1) = r1o;
                M3s_red_matched.r2(end+1) = r2o;
                M3s_red_matched.r3(end+1) = r3o;
                M3s_red_matched.t1(end+1) = mapH(mi0);
                M3s_red_matched.t2(end+1) = mapH(mj0);
                m30 = mod(-mi0 - mj0, Nphi);
                if ~isKey(mapH, m30), continue; end
                M3s_red_matched.t3(end+1) = mapH(m30);
                M3s_red_matched.beta(end+1)  = angle(Bij);
                M3s_red_matched.W3mag(end+1) = abs(Bij);
            end
        else
            [mi0_grid, mj0_grid] = ndgrid(mk0_keep, mk0_keep);
            m30_grid = mod(-mi0_grid - mj0_grid, Nphi);
            valid = ismember(mi0_grid, mk0_keep) & ismember(mj0_grid, mk0_keep) & ismember(m30_grid, mk0_keep);
            [I,J] = find(valid);
            for u = 1:numel(I)
                mi0 = mi0_grid(I(u),J(u)); mj0 = mj0_grid(I(u),J(u)); m30 = m30_grid(I(u),J(u));
                Bij = Bspec(mi0+1, mj0+1);
                M3s_red_matched.r1(end+1) = triplets(t,1)+1;
                M3s_red_matched.r2(end+1) = triplets(t,2)+1;
                M3s_red_matched.r3(end+1) = triplets(t,3)+1;
                M3s_red_matched.t1(end+1) = mapH(mi0);
                M3s_red_matched.t2(end+1) = mapH(mj0);
                M3s_red_matched.t3(end+1) = mapH(m30);
                M3s_red_matched.beta(end+1)  = angle(Bij);
                M3s_red_matched.W3mag(end+1) = abs(Bij);
            end
        end
    end
end

% -------- meta/debug -----------------------------------
dbg = struct();
dbg.Ny_all = Ny_all;
dbg.Ny     = numel(Xbase);
dbg.s_min  = min(double(s_r));
dbg.s_max  = max(double(s_r));
dbg.s_med  = median(double(s_r));
dbg.s_frac_neg = mean(double(s_r) < 0);
dbg.decimator_used = decimator_used;
dbg.decim_energy_before = decimator_used * decim_energy_before;
dbg.decim_energy_after  = decimator_used * decim_energy_after;
dbg.triplet_mode_used = triplet_mode_used;
dbg.nr = nr; dbg.Npairs = Npairs; dbg.Ntrip = Ntrip;

meta = struct('time_total', toc(t_all), 'cfg', cfg, 'max_r', max_r, 'dbg', dbg);

% Exports
if Ntrip > 0
    meta.exports.M3_bspec_complex = toCPU(Bspec_num);   % numerators
    meta.exports.M3_bspec_denom   = D_scalar;           % scalar
    meta.exports.M3_triplets      = triplets;           % 0-based sorted
else
    meta.exports.M3_bspec_complex = complex(zeros(Nphi,Nphi,0));
    meta.exports.M3_bspec_denom   = 0;
    meta.exports.M3_triplets      = zeros(0,3);
end

% -------- integrity prints -----------------------
if cfg.verbose
    fprintf('[se2->so2 CROSS M3+M2] total=%.3fs | rings=%d | pairs=%d | triplets=%d (mode=%s)\n', ...
        meta.time_total, nr, Npairs, Ntrip, triplet_mode_used);
    fprintf('[s(y)] min=%.3e max=%.3e med=%.3e frac(neg)=%.1f%% (Ny=%d / Ny_all=%d)\n', ...
        dbg.s_min, dbg.s_max, dbg.s_med, 100*dbg.s_frac_neg, dbg.Ny, dbg.Ny_all);
    if dbg.decimator_used
        ratio = dbg.decim_energy_after / max(dbg.decim_energy_before,1e-30);
        fprintf('[decimator] energy_before=%.3e energy_after=%.3e ratio=%.3f\n', ...
            dbg.decim_energy_before, dbg.decim_energy_after, ratio);
    else
        fprintf('[decimator] bypassed (Nphi_{os}==Nphi or empty)\n');
    end
    fprintf('[M2_red] data: %dx%d (Δ x pairs) | denom (scalar)=%g\n', ...
        size(M2_red.data,1), size(M2_red.data,2), D_scalar);
    fprintf('[M3_red] data: %dx%dx%d (Δ x Δ x trip) | denom (scalar)=%g\n', ...
        size(M3_red.data,1), size(M3_red.data,2), size(M3_red.data,3), D_scalar);
    fprintf('[EXPORT] M2_dense = [%d x %d x %d] (DIRECT-shaped, normalized)\n', size(M2_dense,1), size(M2_dense,2), size(M2_dense,3));
    Q = numel(M3s_red_matched.beta);
    fprintf('[EXPORT] M3s_red_matched: Q=%d samples | mode=%s\n', Q, M3s_red_matched.mode);
    if Q>0
        sN = min(3,Q);
        fprintf('         sample[1..%d]: (r1,r2,r3|t1,t2,t3|beta|W)\n', sN);
        for s=1:sN
            fprintf('           (%d,%d,%d | %d,%d,%d | %+8.3f | %.3e)\n', ...
                M3s_red_matched.r1(s), M3s_red_matched.r2(s), M3s_red_matched.r3(s), ...
                M3s_red_matched.t1(s), M3s_red_matched.t2(s), M3s_red_matched.t3(s), ...
                M3s_red_matched.beta(s), M3s_red_matched.W3mag(s));
        end
    else
        fprintf('         (no samples)\n');
    end
    fprintf('[EXPORT][norm] match_direct_norm=%d | M2 scale=%g | M3spec sampling=from COMPLEX grid\n', ...
        cfg.match_direct_norm, scale_M2_export);
end

% -------- final sanity -------------------------------------------
assert(isequal(size(M2_num), [nr nr Nphi]), 'M2_num has wrong size (expected [nr nr Nphi]).');
assert(isequal(size(M2_dense), [nr nr Nphi]), 'M2_dense has wrong size.');
% If any M3 samples exist, ensure vector lengths match
fieldsRow = {'r1','r2','r3','t1','t2','t3','beta','W3mag'};
if ~isempty(M3s_red_matched.beta)
    lens = cellfun(@(f) numel(M3s_red_matched.(f)), fieldsRow);
    assert(all(lens==lens(1)), 'M3s_red_matched fields have mismatched lengths.');
end
if ~isempty(mk1_keep)
    Nm = numel(mk1_keep);
    if ~isempty(M3s_red_matched.t1)
        assert(all(M3s_red_matched.t1>=1 & M3s_red_matched.t1<=Nm), 't1 out of range.');
        assert(all(M3s_red_matched.t2>=1 & M3s_red_matched.t2<=Nm), 't2 out of range.');
        assert(all(M3s_red_matched.t3>=1 & M3s_red_matched.t3<=Nm), 't3 out of range.');
    end
end
end

% ====================== helpers =========================
function pairs = all_upper_pairs_0based(max_r)
%ALL_UPPER_PAIRS_0BASED  Return all (r1,r2) with 0-based radii and r1<=r2.
[R1,R2] = find(triu(true(max_r+1),0));
pairs = [R1-1, R2-1];
end

function triplets = build_triplets(cfg, max_r, nr)
%BUILD_TRIPLETS  Construct list of 0-based radius triplets according to cfg.triplets_mode.
mode = lower(cfg.triplets_mode);
switch mode
    case 'explicit'
        T = sort(cfg.triplets_list, 2);
        T = unique(T, 'rows');
        T = T(all(T>=0,2) & all(T<=max_r,2), :);
        triplets = T;
    case 'window3'
        Off = cfg.triplet_offsets;
        assert(~isempty(Off), 'triplets_mode=window3 requires cfg.triplet_offsets');
        T = [];
        for r = 0:max_r
            for j=1:size(Off,1)
                tri = r + Off(j,:);
                if all(tri>=0) && all(tri<=max_r)
                    tri = sort(tri);
                    T(end+1,:) = tri; 
                end
            end
        end
        if ~isempty(T), T = unique(T,'rows'); end
        triplets = T;
    case 'window'
        hop = max(1, cfg.window_hop);
        if max_r >= 2
            ixs = 0:hop:(max_r-2);
            T = [ixs(:), ixs(:)+1, ixs(:)+2];
        else
            T = zeros(0,3);
        end
        triplets = T;
    case 'adjacent3'
        if max_r >= 2
            triplets = [ (0:max_r-2).', (1:max_r-1).', (2:max_r).' ];
        else
            triplets = zeros(0,3);
        end
    case 'all'
        if nr >= 3
            C = nchoosek(0:max_r, 3);
        else
            C = zeros(0,3);
        end
        triplets = C;
    case 'random'
        assert(~isempty(cfg.max_triplets) && isfinite(cfg.max_triplets) && cfg.max_triplets>=1, ...
            'triplets_mode=random requires finite, positive cfg.max_triplets');
        if ~isempty(cfg.triplets_seed), oldRng = rng; rng(cfg.triplets_seed,'twister'); end
        Ttot = nchoosek(nr,3);
        k    = min(cfg.max_triplets, Ttot);
        if Ttot <= 5e5
            C = nchoosek(0:max_r,3);
            if size(C,1) > k
                idx = randperm(size(C,1), k);
                C = C(idx,:);
            end
            triplets = C;
        else
            triplets = sample_random_triplets_0based(nr, k);
        end
        if exist('oldRng','var'), rng(oldRng); end
    otherwise % 'none'
        triplets = zeros(0,3);
end
if isfield(cfg,'max_triplets') && ~isempty(cfg.max_triplets) && size(triplets,1) > cfg.max_triplets
    triplets = triplets(1:cfg.max_triplets,:);
end
end

function T = sample_random_triplets_0based(nr, k)
%SAMPLE_RANDOM_TRIPLETS_0BASED  Sample k unique, sorted 0-based triplets from nr rings.
T = zeros(k,3);
seen = containers.Map('KeyType','char','ValueType','logical');
cnt = 0;
while cnt < k
    x = sort(randperm(nr,3)-1); % 0-based
    key = sprintf('%d_%d_%d', x(1),x(2),x(3));
    if ~isKey(seen, key)
        cnt = cnt + 1;
        T(cnt,:) = x;
        seen(key) = true;
    end
end
end

function H = angular_decimate_os_to_Nphi_gpuaware(Hos_cpu, Nphi_os, Nphi, useGPU)
%ANGULAR_DECIMATE_OS_TO_NPHI_GPUAWARE  Downsample oversampled angular profiles (GPU-aware).
if ~useGPU
    H = angular_decimate_os_to_Nphi(Hos_cpu, Nphi_os, Nphi);
    return;
end
if (Nphi_os == 2*Nphi) && mod(Nphi,2)==0
    Hos = gpuArray(Hos_cpu);
    C_os = fft(Hos, [], 1) / sqrt(Nphi_os);
    Ny = size(Hos,2);
    half = Nphi/2;
    C_lp = zeros(Nphi, Ny, 'like', C_os);
    C_lp(1,:) = C_os(1,:);
    if half >= 2, C_lp(2:half,:) = C_os(2:half,:); end
    C_lp(half+1,:) = 0.5*(C_os(half+1,:) + C_os(Nphi_os - half + 1,:));
    if half >= 2, C_lp(half+2:end,:) = C_os(Nphi_os - (half-2) : Nphi_os, :); end
    H_unit = real(ifft(C_lp, [], 1) * sqrt(Nphi));
    H = sqrt(Nphi / Nphi_os) * H_unit;
    return;
end
H = angular_decimate_os_to_Nphi(Hos_cpu, Nphi_os, Nphi);
H = gpuArray(H);
end

function H = angular_decimate_os_to_Nphi(Hos, Nphi_os, Nphi)
%ANGULAR_DECIMATE_OS_TO_NPHI  Downsample oversampled angular profiles to Nphi.
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
    for j = 1:Ny
        C_lp(:,j) = interp1(m_os, Csh(:,j), m_t, 'linear', 0.0);
    end
    C_lp = ifftshift(C_lp,1);
    H_unit = real(ifft(C_lp, [], 1) * sqrt(Nphi));
end
H = sqrt(Nphi / Nphi_os) * H_unit;
end

function y = ternary(cond,a,b)
%TERNARY  Return a if cond is true, else b.
if cond, y=a; else, y=b; end
end
