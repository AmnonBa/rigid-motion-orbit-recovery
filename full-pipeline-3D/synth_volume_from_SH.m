function V = synth_volume_from_SH(F, Lmax, r_shells, shell_bw, d, mask_rad, shell_cnorm)
% SYNTH_VOLUME_FROM_SH  Reconstruct a 3-D volume from shellwise spherical-harmonic (SH) coefficients.
% --------------------------------------------------------------------------------------------------
%   V = synth_volume_from_SH(F, Lmax, r_shells, shell_bw, d, mask_rad, shell_cnorm)
%
% Reconstructs a real-valued 3-D volume on a cubic grid by evaluating a band-limited
% spherical-harmonic expansion on a set of Gaussian radial “shells”, then summing
% over shells and angular orders inside a spherical mask.
%
% MODEL
%   The volume is represented as
%       f(r,θ,φ) ≈ Σ_{s=1}^R  W_s(r) · Σ_{ℓ=0}^{Lmax} Σ_{m=-ℓ}^{ℓ} F_{ℓm}(s) Y_ℓ^m(θ,φ),
%   where W_s(r) = c_s · exp(−(r − μ_s)^2 / (2 σ^2)) are normalized Gaussian radial
%   windows centered at μ_s = r_shells(s) with common width σ = shell_bw, and
%   c_s = shell_cnorm(s) is the L2(r^2dr)-normalization factor for shell s.
%
% INPUTS
%   F           cell(Lmax+1,1)
%               F{ℓ+1} is a (2ℓ+1)×R matrix of complex SH coefficients on R shells:
%               rows are m = −ℓ:ℓ (in order), columns are shells s = 1..R.
%   Lmax        nonnegative integer, maximum SH degree.
%   r_shells    [1×R] vector of shell centers μ_s in normalized radius units r∈[0,1].
%   shell_bw    scalar σ > 0, Gaussian shell bandwidth in radius units.
%   d           integer, output cubic grid size (volume is d×d×d).
%   mask_rad    scalar in (0,1], spherical support radius (values outside are set NaN).
%   shell_cnorm [1×R] vector of per-shell normalization constants (from construction).
%
% OUTPUT
%   V           d×d×d real array. Voxels outside the spherical mask (radius mask_rad)
%               are set to NaN. Inside the mask, V contains the synthesized volume.
%
% NUMERICAL NOTES
%   • The implementation evaluates Y_ℓ^m(θ,φ) only at voxels inside the spherical mask,
%     caching the Y-basis (“Ycache”) for each ℓ to avoid recomputation across shells.
%   • The sum over m is vectorized as (coeff.' * Ylm) where Ylm stacks Y_ℓ^m rows.
%   • The radial windows are evaluated per voxel and shell: W_s(r) = c_s · exp(…).
%   • The final volume is real by construction; we take real(·) to suppress tiny
%     numerical imaginary parts due to floating-point roundoff.
%
% EXAMPLE
%   % Given previously estimated coefficients (F) and parameters:
%   V = synth_volume_from_SH(F, 4, linspace(0.1,0.95,8), 0.04, 128, 0.9, shell_cnorm);
%
% DEPENDENCIES
%   • sphharmY(l,m,theta,phi) — physics-normalized spherical harmonics (orthonormal on S^2).
%
% --------------------------------------------------------------------------------------------------

    % ----- light input checks (optional but helpful) ---------------------
    if ~(isscalar(Lmax) && Lmax >= 0 && Lmax == floor(Lmax))
        error('Lmax must be a nonnegative integer.');
    end
    R = numel(r_shells);
    if numel(shell_cnorm) ~= R
        error('shell_cnorm must have the same length as r_shells.');
    end
    if ~iscell(F) || numel(F) ~= (Lmax+1)
        error('F must be a cell array of length Lmax+1.');
    end
    for ell = 0:Lmax
        if ~isequal(size(F{ell+1},1), 2*ell+1) || size(F{ell+1},2) ~= R
            error('F{%d} must be of size (2*%d+1) x R.', ell+1, ell);
        end
    end

    % ----- grid & spherical coordinates ----------------------------------
    mL  = arrayfun(@(l) 2*l+1, 0:Lmax);   % row counts per ℓ
    lin = linspace(-1, 1, d);
    [x, y, z] = ndgrid(lin, lin, lin);
    r   = sqrt(x.^2 + y.^2 + z.^2);
    mask = (r <= mask_rad);               % spherical support

    % angles only where we will evaluate (inside mask)
    theta = acos( max(min(z ./ max(r, eps), 1), -1) );   % θ ∈ [0,π]
    phi   = mod(atan2(y, x), 2*pi);                      % φ ∈ [0,2π)
    V     = nan(d, d, d);

    idx = find(mask);                   % linear indices of voxels in mask
    nv  = numel(idx);
    th  = theta(idx);
    ph  = phi(idx);

    % ----- cache spherical harmonics for each ℓ at all masked voxels -----
    Ycache = cell(Lmax+1, 1);
    for ell = 0:Lmax
        m = mL(ell+1);
        ms = (-ell:ell).';
        Ylm = zeros(m, nv);
        for j = 1:m
            % physics-normalized Y_ℓ^{m_j}(θ,φ), returned as row
            Ylm(j, :) = sphharmY(ell, ms(j), th, ph);
        end
        Ycache{ell+1} = Ylm;            % (2ℓ+1) × nv
    end

    % ----- accumulate contribution per shell -----------------------------
    Vmask = zeros(nv, 1);               % values only on masked voxels
    for s = 1:R
        mu = r_shells(s);               % shell center
        cs = shell_cnorm(s);            % shell normalization
        % radial window on masked voxels
        Wr = cs * exp( -0.5 * ((r(idx) - mu) ./ max(shell_bw, 1e-6)).^2 );

        % angular synthesis: Σ_{ℓ,m} F_{ℓm}(s) Y_ℓ^m
        Aang = zeros(nv, 1);
        for ell = 0:Lmax
            Ylm   = Ycache{ell+1};      % (2ℓ+1) × nv
            coeff = F{ell+1}(:, s);     % (2ℓ+1) × 1 (m = −ℓ:ℓ)
            % row-by-matrix: (1×(2ℓ+1)) · ((2ℓ+1)×nv) = 1×nv
            Aang  = Aang + (coeff.' * Ylm).';
        end

        % add shell contribution (take real part to suppress tiny imag)
        Vmask = Vmask + real(Wr .* Aang);
    end

    % ----- write back into full 3-D array (NaN outside mask) -------------
    V(idx) = Vmask;
end
