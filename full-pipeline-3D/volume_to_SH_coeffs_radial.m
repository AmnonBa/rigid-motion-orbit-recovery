function [F_shell, Tabc, shell_cnorm] = volume_to_SH_coeffs_radial( ...
    V, Lmax, r_shells, th, ph, w_ang, Ycache, sigma)
% VOLUME_TO_SH_COEFFS_RADIAL
% ------------------------------------------------------------------------------------
%   [F_shell, Tabc, shell_cnorm] = volume_to_SH_coeffs_radial( ...
%       V, Lmax, r_shells, th, ph, w_ang, Ycache, sigma)
%
% Projects a 3-D volume onto a separable basis consisting of:
%   • Angular part: spherical harmonics Y_ℓ^m(θ,φ) up to degree Lmax
%   • Radial part: Gaussian shells centered at radii r_shells with width sigma
%
% The projection is computed by:
%   1) Sampling the volume on concentric spheres of radii {r_n} (radial quadrature),
%      using an angular quadrature {(θ_k, φ_k), w_ang(k)} on S^2.
%   2) For each radius r_n and each ℓ, accumulating the spherical harmonic transform:
%          F_ℓ^m(r_n) = ∫_{S^2} conj(Y_ℓ^m(θ,φ)) · V(r_n, θ, φ) dΩ
%      approximated by ∑_k conj(Y_ℓ^m(θ_k,φ_k)) · V(r_n,θ_k,φ_k) · w_ang(k).
%   3) Collapsing the radial samples onto the Gaussian shell basis W_s(r) via a
%      weighted inner product in L2(r^2dr).
%
% INPUTS
%   V          : d×d×d real/float; the volume defined on the cube [-1,1]^3 (assumed).
%   Lmax       : nonnegative integer; maximum spherical-harmonic degree.
%   r_shells   : [1×R] shell centers μ_s in radius units r∈[0,1].
%   th, ph     : angular nodes (vectors) for S^2 quadrature (θ∈[0,π], φ∈[0,2π)).
%   w_ang      : [numel(th)*? or same length as angular grid] S^2 quadrature weights.
%   Ycache     : cell(Lmax+1,1), where Ycache{ℓ+1} is (2ℓ+1)×K with rows m=−ℓ:ℓ
%                and columns k indexing (θ_k, φ_k); this is conj(Y_ℓ^m) evaluated
%                on the angular grid (or you may pass Y_ℓ^m and we conjugate here).
%   sigma      : scalar > 0; Gaussian shell bandwidth.
%
% OUTPUTS
%   F_shell    : cell(Lmax+1,1); F_shell{ℓ+1} is (2ℓ+1)×R with columns s=1..R
%                giving F_ℓ^m on each Gaussian shell s.
%   Tabc       : R×R×R tensor of triple radial overlaps
%                  T(a,b,c) = ∫ W_a(r) W_b(r) W_c(r) r^2 dr
%                used to form degree-3 couplings/bispectrum rows.
%   shell_cnorm: [1×R] normalization constants c_s for each shell (so that
%                ||W_s||_{L2(r^2dr)} = 1).
%
% NOTES
%   • Radial quadrature uses a uniform trapezoidal rule over r∈[1e-4,1].
%   • The volume is accessed via griddedInterpolant with cubic interpolation, and
%     ‘nearest’ extrapolation outside the cube.
%   • The Gaussian shell basis and normalization are produced by
%       [W, shell_cnorm] = gaussian_shell_basis(r_nodes, r_shells, sigma, w_nodes).
%     Each row W(s,:) is the (normalized) radial window W_s(r) sampled at r_nodes.
%   • Complexity (dominant): O(nr·K·(Lmax+1)^2), where nr=#radial nodes, K=#angles.
%
% DEPENDENCIES
%   radial_quadrature, gaussian_shell_basis, triple_overlap_tensor
% ------------------------------------------------------------------------------------

    % --- radial quadrature nodes and weights ---
    nr = 512;
    [r_nodes, w_nodes] = radial_quadrature(nr);

    % --- shell basis (normalized in L2(r^2dr)) ---
    R = numel(r_shells);
    mL = arrayfun(@(l) 2*l+1, 0:Lmax);
    [W, cvec] = gaussian_shell_basis(r_nodes, r_shells, sigma, w_nodes);

    % --- precompute unit directions for angular nodes (K = numel(th) = numel(ph)) ---
    st = sin(th(:));  ct = cos(th(:));
    cp = cos(ph(:));  sp = sin(ph(:));
    dirs = [st.*cp, st.*sp, ct];  % K×3

    % --- 3D interpolant of volume on [-1,1]^3 ---
    d = size(V,1);
    lin = linspace(-1, 1, d);
    Fint = griddedInterpolant({lin, lin, lin}, double(V), 'cubic', 'nearest');

    % --- accumulate SH coefficients on raw radial nodes F_ℓ^m(r_n) ---
    F_r = cell(Lmax+1, 1);
    for ell = 0:Lmax
        F_r{ell+1} = zeros(mL(ell+1), numel(r_nodes));  % (2ℓ+1)×nr
    end

    rw = (r_nodes.^2) .* w_nodes;  % weight r^2 dr for later projection
    for n = 1:numel(r_nodes)
        r = r_nodes(n);
        XYZ  = r * dirs;                               % K×3 points on sphere r
        vals = Fint(XYZ(:,1), XYZ(:,2), XYZ(:,3));     % V(r,θ_k,φ_k) samples
        for ell = 0:Lmax
            % Ycache{ell+1}: (2ℓ+1)×K, already evaluated at (θ_k,φ_k)
            Ylm = Ycache{ell+1};
            % F_ℓ^m(r_n) = ∑_k conj(Y_ℓ^m(θ_k,φ_k)) V(r_n,θ_k,φ_k) w_ang(k)
            % (If Ycache stores Y_ℓ^m, then conj(Ylm) is correct as below.)
            F_r{ell+1}(:, n) = conj(Ylm) * (w_ang(:) .* vals(:));
        end
    end

    % --- collapse radial nodes onto Gaussian shells with L2(r^2dr) weights ---
    F_shell = cell(Lmax+1, 1);
    for ell = 0:Lmax
        % include r^2 dr weights prior to projecting onto W
        Fr_weighted      = F_r{ell+1} .* rw.';  % (2ℓ+1)×nr
        F_shell{ell+1}   = Fr_weighted * W.';   % (2ℓ+1)×R
    end

    % --- triple radial overlap tensor T(a,b,c) for degree-3 couplings ---
    Tabc = triple_overlap_tensor(W, r_nodes, w_nodes);

    % --- per-shell normalization constants (returned for synthesis, diagnostics) ---
    shell_cnorm = cvec;
end


function T = triple_overlap_tensor(W, r_nodes, w_nodes)
% TRIPLE_OVERLAP_TENSOR
% -------------------------------------------------------------------------
%   T = triple_overlap_tensor(W, r_nodes, w_nodes)
%
% Computes the 3-way radial overlap tensor of normalized Gaussian shells:
%     T(a,b,c) = ∫ W_a(r) W_b(r) W_c(r) r^2 dr
% using the same radial grid and weights used to build W.
%
% INPUTS
%   W        : R×nr matrix; row s is the (normalized) radial window W_s(r_n).
%   r_nodes  : [nr×1] radial nodes (in [0,1]).
%   w_nodes  : [nr×1] radial quadrature weights dr (trapezoid rule).
%
% OUTPUT
%   T        : R×R×R tensor of triple overlaps.
% -------------------------------------------------------------------------
    R  = size(W, 1);
    rw = (r_nodes.^2) .* w_nodes;    % weight for L2(r^2 dr)
    T  = zeros(R, R, R);

    for a = 1:R
        Wa = W(a, :);
        for b = 1:R
            Wab = Wa .* W(b, :);     % elementwise product on the radial grid
            for c = 1:R
                % Discrete approximation of ∫ W_a W_b W_c r^2 dr
                T(a,b,c) = sum( (Wab .* W(c, :)) .* rw.' );
            end
        end
    end
end
