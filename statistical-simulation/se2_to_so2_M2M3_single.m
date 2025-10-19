function out = se2_to_so2_M2M3_single(f, cfg, target)
% SE2_TO_SO2_M2M3_SINGLE
% ---------------------------------------------------------------
% Single-value SE(2)->SO(2) reduction:
%   - M^2 at (r1,r2,phi)
%   - M^3 at (r1,r2,r3, dphi1, dphi2)
% Uses ONE global denominator D_scalar = sum_y s(y).
% Unit steps align with your v5 normalization (unitary ring FFT).
% ---------------------------------------------------------------

% ---- Prepare translations, s(y), denominator, indices ----
[Xbase, Ybase, s_r, D_scalar, IDX1D, ~] = local_prepare_common_bits(f, cfg);

% Keep active translations
idx_active = find(s_r ~= 0);
if isempty(idx_active)
    out = struct('D_scalar', D_scalar, 'M2_val', 0, 'M3_val', 0);
    return;
end
sa = s_r(idx_active);

% Rings we need
need_r = [];
if isfield(target,'M2') && ~isempty(target.M2)
    need_r = [need_r, target.M2.r1, target.M2.r2];
end
if isfield(target,'M3') && ~isempty(target.M3)
    need_r = [need_r, target.M3.r1, target.M3.r2, target.M3.r3];
end
need_r = unique(need_r(:).');

% Build spectra for needed rings (unitary)
fCPU = double(f);
rings = cell(1, max(need_r)+1);
for r0 = need_r
    rings{r0+1} = local_ring_spectrum_one(fCPU, Xbase(idx_active), Ybase(idx_active), r0, cfg);
end

% Initialize outputs
out = struct('D_scalar', D_scalar, 'M2_val', [], 'M3_val', []);

% ---- Single M^2 ----
if isfield(target,'M2') && ~isempty(target.M2)
    Nphi = cfg.Nphi;
    r1 = target.M2.r1; r2 = target.M2.r2; phi = target.M2.phi;
    C1  = rings{r1+1};                     % [Nphi x Ny_active]
    C2n = rings{r2+1}(IDX1D,:);            % -m rows
    Sspec = (C1 .* C2n) * sa;              % [Nphi x 1]
    m = (0:Nphi-1).';
    M2_val = real( (exp(1i*m*phi).') * Sspec ) / Nphi;   % correct 1/Nphi scaling
    if abs(D_scalar) > 1e-30, M2_val = M2_val / D_scalar; else, M2_val = 0; end
    out.M2_val = M2_val;
end

% ---- Single M^3 ----
if isfield(target,'M3') && ~isempty(target.M3)
    Nphi = cfg.Nphi;
    r1 = target.M3.r1; r2 = target.M3.r2; r3 = target.M3.r3;
    d1 = target.M3.dphi1; d2 = target.M3.dphi2;

    % Ring spectra (Nphi x Ny_active)
    C1 = rings{r1+1};
    C2 = rings{r2+1};
    C3 = rings{r3+1};

    % Exponentials for the two evaluation angles
    m  = (0:Nphi-1).';
    e1 = exp(1i*m*d1);
    e2 = exp(1i*m*d2);

    % Preweight ring spectra by those exponentials
    A      = e1 .* C1;          % Nphi x Ny
    B      = e2 .* C2;          % Nphi x Ny
    Brev   = B(IDX1D,:);        % b_{-k}
    C3neg  = C3(IDX1D,:);       % c_k = c_{-k} (i.e. C3[-k])

    % Optional: GPU (keeps API identical)
    useGPU = isfield(cfg,'useGPU') && cfg.useGPU && parallel.gpu.GPUDevice.isAvailable;
    if useGPU
        A = gpuArray(A); Brev = gpuArray(Brev); C3neg = gpuArray(C3neg);
        sa_g = gpuArray(sa);
    else
        sa_g = sa;
    end

    % Compute circular convolution z = Brev ⊛ C3neg for ALL translations
    % via 1-D FFTs along the harmonic axis (dimension 1)
    FB = fft(Brev,  [], 1);     % Nphi x Ny
    FC = fft(C3neg, [], 1);     % Nphi x Ny
    Z  = ifft(FB .* FC, [], 1); % Nphi x Ny, each col z_m1 = sum_m2 b_{m2} c_{m1+m2}

    % For each translation y: T_y = sum_m1 A(m1,y) * Z(m1,y)
    Ty = real(sum(A .* Z, 1));  % 1 x Ny

    % Weight by s(y) and finish the same normalization as before
    M3_num = Ty * sa_g;         % scalar
    M3_val = (M3_num / Nphi);
    if abs(D_scalar) > 1e-30, M3_val = M3_val / D_scalar; else, M3_val = 0; end

    % Keep your extra 1/sqrt(Nphi) to match your baselines
    out.M3_val = gather(M3_val) / sqrt(Nphi);
end
end

% ====================== local helpers ===========================
function [Xbase, Ybase, s_r, D_scalar, IDX1D, IDX2D_flat] = local_prepare_common_bits(f, cfg)
S = size(f,1); ctr = (S+1)/2; fCPU = double(f);
Rdisk = cfg.R; Nphi = cfg.Nphi;

% translations base set
[Xg,Yg] = meshgrid(1:S, 1:S);
rr_full = hypot(Xg-ctr, Yg-ctr);
if isfield(cfg,'limit_translations') && cfg.limit_translations
    Rbase = min(max(cfg.trans_radius,0), Rdisk - 1e-9);
else
    Rbase = Rdisk - 1e-9;
end
base_valid = rr_full <= Rbase;
[y_list, x_list] = find(base_valid);
Xbase = double(x_list); Ybase = double(y_list);

% boundary antipodal factor
if isfield(cfg,'Ntheta') && ~isempty(cfg.Ntheta), Ntheta = cfg.Ntheta; else, Ntheta = max(180, round(8*pi*Rdisk)); end
bnd_R = Rdisk - cfg.delta_pix;
theta_SE = linspace(0, 2*pi*(1-1/Ntheta), Ntheta);
U0 = [bnd_R*cos(theta_SE); bnd_R*sin(theta_SE)]; U1 = -U0;
U0 = cat(3,U0(1,:),U0(2,:));  U1 = cat(3,U1(1,:),U1(2,:));
v0 = interp2(fCPU, Xbase + U0(:,:,1), Ybase + U0(:,:,2), cfg.bndMethod, 0.0);
v1 = interp2(fCPU, Xbase + U1(:,:,1), Ybase + U1(:,:,2), cfg.bndMethod, 0.0);
s_r = mean(v0 .* v1, 2);
D_scalar = sum(s_r);

% indices for -m and -(m1+m2)
[m1g,m2g] = ndgrid(0:Nphi-1, 0:Nphi-1);
IDX2D = mod(-(m1g + m2g), Nphi) + 1;
IDX1D = mod(-(0:Nphi-1), Nphi) + 1;
IDX2D_flat = IDX2D(:);
end

function C_r = local_ring_spectrum_one(fCPU, Xbase_act, Ybase_act, r0, cfg)
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
C_r = fft(Hos, [], 1) / sqrt(Nphi);                    % unitary spectra
end

function H = angular_decimate_os_to_Nphi(Hos, Nphi_os, Nphi)
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
if cond, y=a; else, y=b; end
end
