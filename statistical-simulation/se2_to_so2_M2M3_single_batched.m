function out = se2_to_so2_M2M3_single_batched(f, cfg, target)
% SE2_TO_SO2_M2M3_SINGLE_BATCHED
% -------------------------------------------------------------------------
% Batched version of single-value SE(2)->SO(2) reduction:
%   - Accepts f of size S x S x B (B images)
%   - Computes, per image, a SINGLE M^2(r1,r2,phi) and a SINGLE M^3(r1,r2,r3,dphi1,dphi2)
%   - Uses ONE global denominator per image: D_scalar(b) = sum_y s_b(y).
% GPU-first: vectorized boundary/ring sampling and numerators across batch.
% Matches your v5 unitary conventions (ring FFTs scaled by 1/sqrt(Nphi)).
% -------------------------------------------------------------------------

%% ------------------------- Setup & device -------------------------
assert(ndims(f) >= 2 && ndims(f) <= 3, 'f must be SxS or SxSxB');
[S1,S2,B] = size(f);
assert(S1==S2, 'f must be square per page');
S = S1;

useGPU = isfield(cfg,'useGPU') && cfg.useGPU && parallel.gpu.GPUDevice.isAvailable;
bndMethod  = pickOrDefault(cfg, 'bndMethod',  'linear');   % for boundary interp2
ringMethod = pickOrDefault(cfg, 'ringMethod', 'linear');   % for ring interp2
Rdisk      = cfg.R;
Nphi       = cfg.Nphi;

if useGPU
    Fbig = gpuArray( reshape(double(f), S, S*B) );   % S x (S*B)
else
    Fbig = reshape(double(f), S, S*B);
end

ctr = (S+1)/2;
[Xg,Yg] = meshgrid(1:S, 1:S);
rr_full = hypot(Xg-ctr, Yg-ctr);

% Base translations (shared across batch)
if isfield(cfg,'limit_translations') && cfg.limit_translations
    Rbase = min(max(cfg.trans_radius,0), Rdisk - 1e-9);
else
    Rbase = Rdisk - 1e-9;
end
base_valid = rr_full <= Rbase;
[y_list, x_list] = find(base_valid);
Xbase = double(x_list(:));   % Ny_full x 1
Ybase = double(y_list(:));   % Ny_full x 1
Ny_full = numel(Xbase);

% Active translations are determined per reduction via s(y)≠0, but we first
% compute s(y) for ALL y; later we prune consistently across the batch.
% Boundary directions:
if isfield(cfg,'Ntheta') && ~isempty(cfg.Ntheta)
    Ntheta = cfg.Ntheta;
else
    Ntheta = max(180, round(8*pi*Rdisk));
end
bnd_R   = Rdisk - cfg.delta_pix;
thetaSE = linspace(0, 2*pi*(1-1/Ntheta), Ntheta);
U0x = bnd_R * cos(thetaSE);   U0y = bnd_R * sin(thetaSE);
U1x = -U0x;                   U1y = -U0y;

% Precompute tiled offsets (PAGE-MAJOR) to address pages within Fbig’s columns
% Correct mapping for B>1:
if useGPU
    page_idx_row_full = gpuArray(kron(0:B-1, ones(1, Ny_full)));  % [0..B-1] blocks, each length Ny_full
else
    page_idx_row_full = kron(0:B-1, ones(1, Ny_full));
end
offset_row_full = S * page_idx_row_full;                          % 1 x (Ny_full*B)

%% ------------------------- s(y) and D_scalar(b) -------------------------
% Sample boundaries for all y, all theta, all images in one go
% Build (Ntheta x Ny_full) base coords, then replicate across pages with column offsets.
Xb0 = (Xbase(:).') + U0x(:);   % Ntheta x Ny_full
Yb0 = (Ybase(:).') + U0y(:);
Xb1 = (Xbase(:).') + U1x(:);
Yb1 = (Ybase(:).') + U1y(:);

% Make them Ntheta x (Ny_full*B) with per-page column offsets in X:
if useGPU
    oneNt = gpuArray.ones(Ntheta,1);
else
    oneNt = ones(Ntheta,1);
end
Xb0_rep = repmat(Xb0, 1, B) + oneNt * offset_row_full;
Yb0_rep = repmat(Yb0, 1, B);
Xb1_rep = repmat(Xb1, 1, B) + oneNt * offset_row_full;
Yb1_rep = repmat(Yb1, 1, B);

v0 = interp2(Fbig, Xb0_rep, Yb0_rep, bndMethod, 0.0);   % Ntheta x (Ny_full*B)
v1 = interp2(Fbig, Xb1_rep, Yb1_rep, bndMethod, 0.0);

% s(y) per image: mean over theta of v0 .* v1
s_prod = v0 .* v1;                                       % Ntheta x (Ny_full*B)
s_mean = mean(s_prod, 1);                                % 1 x (Ny_full*B)
s_r_all = reshape(s_mean, 1, Ny_full, B);                % 1 x Ny_full x B
s_r_all = permute(s_r_all, [2 1 3]);                     % Ny_full x 1 x B

% Active translations mask shared across batch (any image with nonzero s)
active_any = any( abs(gather(s_r_all)) > 0, 3 );         % Ny_full x 1 (CPU mask)
idx_active = find(active_any);
if isempty(idx_active)
    out = struct('D_scalar', zeros(B,1), 'M2_val', zeros(B,1), 'M3_val', zeros(B,1));
    return;
end

% Prune to active translations
Ny = numel(idx_active);
s_r = s_r_all(idx_active,1,:);                           % Ny x 1 x B
D_scalar = squeeze(sum(s_r, 1));                         % 1 x 1 x B -> 1 x B
D_scalar = reshape(D_scalar, B, 1);                      % B x 1   (FIXED SHAPE)

% Convenience: a per-page (Ny x 1 x B) weight for pagemtimes
sa = s_r;                                                % Ny x 1 x B

%% ------------------------- Targets & rings needed -------------------------
need_r = [];
if isfield(target,'M2') && ~isempty(target.M2)
    need_r = [need_r, target.M2.r1, target.M2.r2];
end
if isfield(target,'M3') && ~isempty(target.M3)
    need_r = [need_r, target.M3.r1, target.M3.r2, target.M3.r3];
end
need_r = unique(need_r(:).');
max_r  = max([need_r, 0]);

% Index map for -m
IDX1D = mod(-(0:Nphi-1), Nphi) + 1;                      % 1 x Nphi
if useGPU
    IDX1D = gpuArray(IDX1D);
end

%% ------------------------- Ring sampling for ALL images -------------------
% We sample rings at (phi_os) then decimate to Nphi if requested, per ring.
rings = cell(1, max_r+1);

% Build an ACTIVE offset vector of length Ny*B (page-major) for ring sampling
% This ensures correct page alignment even when Ny < Ny_full (after pruning).
if useGPU
    page_idx_row_act = gpuArray(kron(0:B-1, ones(1, Ny)));
else
    page_idx_row_act = kron(0:B-1, ones(1, Ny));
end
offset_row_act = S * page_idx_row_act;                   % 1 x (Ny*B)

for r0 = need_r
    % Oversampling (optional)
    if isfield(cfg,'Nphi_os') && ~isempty(cfg.Nphi_os) && (cfg.Nphi_os > Nphi)
        Nphi_os = cfg.Nphi_os;
    else
        Nphi_os = Nphi;
    end
    phi_os = linspace(0, 2*pi*(1-1/Nphi_os), Nphi_os);
    Vr_x = r0 * cos(phi_os(:));       % Nphi_os x 1
    Vr_y = r0 * sin(phi_os(:));

    % Base coordinates for ACTIVE y’s only
    Xv = Vr_x + (Xbase(idx_active).');    % Nphi_os x Ny
    Yv = Vr_y + (Ybase(idx_active).');    % Nphi_os x Ny

    % Replicate across pages with column offset in X (use ACTIVE offsets)
    if useGPU
        oneNos = gpuArray.ones(Nphi_os,1);
    else
        oneNos = ones(Nphi_os,1);
    end
    Xv_rep = repmat(Xv, 1, B) + oneNos * offset_row_act;   % Nphi_os x (Ny*B)
    Yv_rep = repmat(Yv, 1, B);

    Hos = interp2(Fbig, Xv_rep, Yv_rep, ringMethod, 0.0);  % Nphi_os x (Ny*B)
    Hos = reshape(Hos, Nphi_os, Ny, B);                    % Nphi_os x Ny x B

    % Decimate to Nphi (if needed)
    if Nphi_os ~= Nphi
        H = angular_decimate_os_to_Nphi_batched(Hos, Nphi_os, Nphi);
    else
        H = Hos;
    end

    % Unitary spectra along harmonic dimension (dim=1): Nphi x Ny x B
    C_r = fft(H, [], 1) / sqrt(Nphi);
    rings{r0+1} = C_r;
end

%% ------------------------- Outputs container ------------------------------
out = struct('D_scalar', gatherIfNeeded(D_scalar', useGPU), ...
             'M2_val',   [], ...
             'M3_val',   []);

%% ------------------------- Single M^2 per image ---------------------------
if isfield(target,'M2') && ~isempty(target.M2)
    r1 = target.M2.r1; r2 = target.M2.r2; phi = target.M2.phi;

    C1  = rings{r1+1};                       % Nphi x Ny x B
    C2  = rings{r2+1};                       % Nphi x Ny x B
    C2n = C2(IDX1D,:,:);                     % -m rows: Nphi x Ny x B

    % Per-page vector: Sspec(:,1,b) = (C1.*C2n) * sa(:,1,b)
    A = C1 .* C2n;                           % Nphi x Ny x B
    Sspec = pagemtimes(A, sa);               % (Nphi x 1 x B)

    % M2: <e^{i m phi}, Sspec> / Nphi / D_scalar(b)
    m  = (0:Nphi-1).';
    if useGPU, m = gpuArray(m); end
    ephi = exp(1i*m*phi);                    % Nphi x 1

    % Dot per page -> 1 x B, then B x 1
    M2_num = squeeze(sum(bsxfun(@times, Sspec, reshape((ephi),[],1,1)), 1)); % 1 x B
    M2_val = (M2_num.'/Nphi);                % B x 1
    % Normalize by D_scalar(b)
    %tiny = 1e-30;
    %mask = abs(out.D_scalar) > tiny;
    %M2_val(~mask) = 0;
    %M2_val(mask)  = real(M2_val(mask))' ./ out.D_scalar(mask);

    out.M2_val = gatherIfNeeded(real(M2_val), useGPU);
end

%% ------------------------- Single M^3 per image ---------------------------
if isfield(target,'M3') && ~isempty(target.M3)
    r1 = target.M3.r1; r2 = target.M3.r2; r3 = target.M3.r3;
    d1 = target.M3.dphi1; d2 = target.M3.dphi2;

    C1 = rings{r1+1};    % Nphi x Ny x B
    C2 = rings{r2+1};    % Nphi x Ny x B
    C3 = rings{r3+1};    % Nphi x Ny x B

    m  = (0:Nphi-1).';
    if useGPU, m = gpuArray(m); end
    e1 = exp(1i*m*d1);   % Nphi x 1
    e2 = exp(1i*m*d2);   % Nphi x 1

    % Preweight by exponentials (broadcast over Ny,B)
    A     = bsxfun(@times, C1, reshape(e1,[],1,1));       % Nphi x Ny x B
    B2    = bsxfun(@times, C2, reshape(e2,[],1,1));       % Nphi x Ny x B
    Brev  = B2(IDX1D,:,:);                                % Nphi x Ny x B
    C3neg = C3(IDX1D,:,:);                                % Nphi x Ny x B

    % Circular convolution along harmonic dim via FFT per page:
    FB = fft(Brev,  [], 1);
    FC = fft(C3neg, [], 1);
    Z  = ifft(conj(FB) .* (FC), [], 1);                           % Nphi x Ny x B

    % Ty(1,1,b) = sum_m A(m,:,b) .* Z(m,:,b)
    Ty = real(squeeze(sum(A .* Z, 1)));                         % Ny x B

    % Final aggregation scalar per image (B x 1)   (FIXED SHAPE)
    Apm = permute(reshape(Ty, [Ny, 1, B]), [2 1 3]);  % 1 x Ny x B
    Bpm    = sa;                                          % Ny x 1 x B
    M3_num = pagemtimes(Apm, Bpm);                        % 1 x 1 x B
    M3_val = reshape(real(M3_num), 1, B) / Nphi /sqrt(Nphi);         % B x 1

    % Normalize by D_scalar(b) and extra 1/sqrt(Nphi)
    %tiny = 1e-30;
    %mask = abs(out.D_scalar) > tiny;
    %M3_val(~mask) = 0;
    %M3_val(mask)  = M3_val(mask) ./ out.D_scalar(mask);
    %M3_val = M3_val / sqrt(Nphi);

    out.M3_val = gatherIfNeeded(M3_val, useGPU);
end

end % main function

% ====================== helpers (batched) ==========================
function v = pickOrDefault(cfg, field, vDefault)
    if isfield(cfg, field) && ~isempty(cfg.(field))
        v = cfg.(field);
    else
        v = vDefault;
    end
end

function X = gatherIfNeeded(X, useGPU)
    if useGPU, X = gather(X); end
end

function H = angular_decimate_os_to_Nphi_batched(Hos, Nphi_os, Nphi)
% HOS: Nphi_os x Ny x B
% Returns H: Nphi x Ny x B (band-limited interpolation in harmonic domain)
    [~,Ny,B] = size(Hos);
    C_os = fft(Hos, [], 1) / sqrt(Nphi_os);      % Nphi_os x Ny x B

    if Nphi_os == 2*Nphi && mod(Nphi,2)==0
        % Optimized half-band copy (keeps unitary convention)
        half = Nphi/2;
        C_lp = zeros(Nphi, Ny, B, 'like', C_os);

        % DC
        C_lp(1,:,:) = C_os(1,:,:);

        if half >= 2
            C_lp(2:half,:,:) = C_os(2:half,:,:);
        end

        % Nyquist (averaged fold)
        C_lp(half+1,:,:) = 0.5*(C_os(half+1,:,:) + C_os(Nphi_os - half + 1,:,:));

        if half >= 2
            C_lp(half+2:end,:,:) = C_os(Nphi_os - (half-2) : Nphi_os, :, :);
        end

        H = real(ifft(C_lp, [], 1) * sqrt(Nphi));   % Nphi x Ny x B
        H = sqrt(Nphi / Nphi_os) * H;               % energy match
        return
    end

    % General SHO band-limit via linear interp on fftshifted indices
    m_os = fftshift((-floor(Nphi_os/2)):(ceil(Nphi_os/2)-1)).';
    m_t  = fftshift((-floor(Nphi/2)) :(ceil(Nphi/2)-1)).';
    Csh  = fftshift(C_os,1);
    C_lp = zeros(Nphi, Ny, B, 'like', C_os);

    for b = 1:B
        for j = 1:Ny
            C_lp(:,j,b) = interp1(m_os, Csh(:,j,b), m_t, 'linear', 0.0);
        end
    end

    C_lp = ifftshift(C_lp,1);
    H = real(ifft(C_lp, [], 1) * sqrt(Nphi));      % unitary ifft
    H = sqrt(Nphi / Nphi_os) * H;
end
