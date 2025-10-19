function [G2_BW, Bisys_BW] = build_BW_invariants_from_translations( ...
    Vpad, Lmax, R, r_shells, shell_bw, th, ph, w_ang, Ycache, ...
    Gaunt, Tabc, plan, Tvox, delta_boundary, mask_radius, row_normalize, verbose)
% BUILD_BW_INVARIANTS_FROM_TRANSLATIONS
% ------------------------------------------------------------------------------
% Boundary-weighted construction of SO(3) invariants (G2 and Bisys) from a
% *single* 3-D volume by averaging over a grid of translations.
%
% For each translation t in the user-supplied list Tvox:
%   1) Translate the (padded) volume: V_t(x) = Vpad(x − t).
%   2) Compute a boundary weight s_δ(t) by correlating nearly-antipodal
%      samples on a sphere of radius ≈ mask_radius*(1−δ). (See helper below.)
%   3) Project V_t onto spherical shells → per-(ℓ,c) SH coefficients F_t.
%   4) Accumulate s_δ(t)·G2_t and s_δ(t)·{Bisys rows, rhs} using a *fixed*
%      sampling plan `plan{ℓ3+1,c}` that lists (l1,l2,a,b) tuples.
% Finally, normalize all accumulators by Σ_t s_δ(t) (and optionally row-normalize).
%
% INPUTS
%   Vpad            : (d×d×d) real/double, *already padded* 3-D volume on grid [-1,1]^3.
%   Lmax            : max SH degree (non-negative integer).
%   R               : number of radial shells.
%   r_shells        : 1×R vector of shell centers (in normalized radius units).
%   shell_bw        : Gaussian shell bandwidth (same units as r_shells).
%   th, ph, w_ang   : S^2 quadrature nodes θ, φ and weights (e.g., Fibonacci sphere).
%   Ycache          : precomputed Y_l^m(θ,φ) blocks up to Lmax (cell, size Lmax+1).
%   Gaunt           : Gaunt tables, Gaunt{l1+1,l2+1,ℓ3+1}.
%   Tabc            : triple radial overlap tensor, T(a,b,c), size R×R×R.
%   plan            : cell(Lmax+1,R); plan{ℓ3+1,c} is N×4 [l1 l2 a b] rows, fixed across t.
%   Tvox            : (#T×3) integer translations [dx dy dz] in *voxels*.
%   delta_boundary  : δ∈(0,1); boundary thinning factor for s_δ(t).
%   mask_radius     : sphere radius (≤1) for boundary sampling.
%   row_normalize   : logical; normalize Bisys rows AFTER averaging (recommended).
%   verbose         : (optional) 0/1/2; if 2 prints per-translation diagnostics.
%
% OUTPUTS
%   G2_BW           : cell(Lmax+1,1), each R×R. Boundary-weighted degree-2 invariants.
%   Bisys_BW        : struct(Lmax+1, R) with fields:
%                       .A : (#rows × (2ℓ3+1)) left-hand rows (averaged & normalized)
%                       .b : (#rows × 1)       right-hand side (averaged & normalized)
%
% NOTES
%   • Using the *same* `plan` across translations enables row-wise comparison
%     with oracle Bisys assembled from the same plan.
%   • Row normalization (after averaging) improves numerical stability when
%     the row magnitudes vary across translations.
%   • The projector `volume_to_SH_coeffs_radial` must match the one used for
%     oracle/synthetic construction (same shells and bandwidth).
%
% SEE ALSO
%   compute_sdelta_on_volume, volume_to_SH_coeffs_radial,
%   build_bisys_sampling_plan, assemble_bisys_oracle_from_plan
% ------------------------------------------------------------------------------

    if nargin < 16 || isempty(verbose), verbose = 1; end

    % --- precompute (2ℓ+1) for reference (not directly used further)
    %mL = arrayfun(@(l) 2*l+1, 0:Lmax); 

    % ===================== Initialize accumulators ======================
    % G2: one R×R Gram per ℓ
    G2_acc = cell(Lmax+1,1);
    for ell = 0:Lmax
        G2_acc{ell+1} = zeros(R, R);
    end

    % Bisys: per (ℓ3,c) block, pre-size using the fixed plan
    Bisys_acc = repmat(struct('A',[],'b',[]), Lmax+1, R);
    for ell = 0:Lmax
        for c = 1:R
            N = size(plan{ell+1, c}, 1);
            Bisys_acc(ell+1, c).A = zeros(N, 2*ell+1);
            Bisys_acc(ell+1, c).b = zeros(N, 1);
        end
    end

    % Σ_t s_δ(t) for later normalization
    Ssum = 0;

    % ==================== Sweep over translations t ====================
    nT = size(Tvox, 1);
    for it = 1:nT
        dv = Tvox(it, :);  % [dx, dy, dz] in voxels

        % (1) Translate volume (content moves; domain unchanged)
        Vt = imtranslate(Vpad, dv, 'cubic', 'OutputView', 'same', 'FillValues', 0);

        % (2) Boundary weight s_δ(t) by near-antipodal correlation on S^2
        sdel = compute_sdelta_on_volume(Vt, delta_boundary, mask_radius, th, ph, w_ang);
        if ~isfinite(sdel) || sdel <= 0
            if verbose > 1
                fprintf('  [BW] t=%3d/%3d | sδ<=0 (skip)\n', it, nT);
            end
            continue;
        end

        % (3) Project V_t → shell SH coefficients
        %     F_t{ℓ+1} is (2ℓ+1)×R; column c is sℓ,c
        [F_t, ~, ~] = volume_to_SH_coeffs_radial(Vt, Lmax, r_shells, th, ph, w_ang, Ycache, shell_bw);

        % (4a) Accumulate s_δ(t)·G2_t for each ℓ
        for ell = 0:Lmax
            G2_t = F_t{ell+1}' * F_t{ell+1};  % R×R
            G2_acc{ell+1} = G2_acc{ell+1} + sdel * G2_t;
        end

        % (4b) Accumulate s_δ(t)·{Bisys rows, rhs} using the fixed plan
        for ell3 = 0:Lmax
            for c = 1:R
                rows = plan{ell3+1, c};     % N×4 : [l1, l2, a, b]
                N = size(rows, 1);
                if N == 0, continue; end

                s3 = F_t{ell3+1}(:, c);     % (2ℓ3+1)×1

                for n = 1:N
                    l1 = rows(n,1);  l2 = rows(n,2);
                    a  = rows(n,3);  b  = rows(n,4);

                    f1 = F_t{l1+1}(:, a);   if ~any(f1), continue; end
                    f2 = F_t{l2+1}(:, b);   if ~any(f2), continue; end

                    Gmap = Gaunt{l1+1, l2+1, ell3+1};
                    if isempty(Gmap) || ~any(Gmap(:)), continue; end

                    % alpha = G_{l1,l2→ℓ3} * vec(f1 * f2^T)  (m3×1)
                    v12   = reshape(f1 * (f2.'), [], 1);
                    alpha = Gmap * v12;
                    if ~any(alpha), continue; end

                    tau   = Tabc(a, b, c);           % scalar T(a,b,c)
                    rowv  = (tau * alpha).';         % 1×(2ℓ3+1)
                    rhs   =  tau * (alpha.' * s3);   % scalar

                    % Weighted accumulation
                    Bisys_acc(ell3+1, c).A(n,:) = Bisys_acc(ell3+1, c).A(n,:) + sdel * rowv;
                    Bisys_acc(ell3+1, c).b(n)   = Bisys_acc(ell3+1, c).b(n)   + sdel * rhs;
                end
            end
        end

        Ssum = Ssum + sdel;
        if verbose > 1
            fprintf('  [BW] t=%3d/%3d | sδ=%.3e\n', it, nT, sdel);
        end
    end

    % =========================== Normalize ============================
    % Divide by Σ_t s_δ(t); ensure stability via max(Ssum, eps)
    G2_BW = cell(Lmax+1,1);
    for ell = 0:Lmax
        G2_BW{ell+1} = G2_acc{ell+1} / max(Ssum, eps);
    end

    Bisys_BW = repmat(struct('A',[],'b',[]), Lmax+1, R);
    for ell3 = 0:Lmax
        for c = 1:R
            A = Bisys_acc(ell3+1, c).A / max(Ssum, eps);
            b = Bisys_acc(ell3+1, c).b / max(Ssum, eps);

            % Optional: row-normalize AFTER averaging (recommended)
            if row_normalize && ~isempty(A)
                rn = sqrt(sum(abs(A).^2, 2));     % row norms
                z  = rn > 0;
                A(z,:) = A(z,:) ./ rn(z);
                b(z)   = b(z)   ./ rn(z);
            end

            Bisys_BW(ell3+1, c).A = A;
            Bisys_BW(ell3+1, c).b = b;
        end
    end
end


function sdel = compute_sdelta_on_volume(V, delta_boundary, mask_radius, th, ph, w_ang)
% COMPUTE_SDELTA_ON_VOLUME
% ------------------------------------------------------------------------------
% Compute boundary weight s_δ(t) for a *translated* volume V by correlating
% nearly-antipodal points on a sphere of radius r0 = mask_radius*(1−δ).
%
%   s_δ(t) = ∫_{S^2}  V(+r0·θ) · V(−r0·θ)  dσ(θ)
%
% Discretization uses the same (θ,φ) nodes and weights (th, ph, w_ang) as the
% projector, ensuring consistent angular quadrature.
%
% INPUTS
%   V              : (d×d×d) translated volume (double).
%   delta_boundary : δ∈(0,1), small positive thinning factor (e.g., 1e−2).
%   mask_radius    : radius ≤ 1 used for the boundary sphere.
%   th, ph         : angular nodes on S^2 (vectors).
%   w_ang          : corresponding quadrature weights (vector, sum ≈ 4π).
%
% OUTPUT
%   sdel           : scalar boundary weight (non-negative in typical cases).
% ------------------------------------------------------------------------------

    % Ensure numeric type accepted by griddedInterpolant
    V = double(V);

    d   = size(V,1);
    lin = linspace(-1, 1, d);

    % 3-D cubic interpolation; outside the grid use nearest extrapolation
    Fint = griddedInterpolant({lin, lin, lin}, V, 'cubic', 'nearest');

    % Unit directions on S^2
    st = sin(th(:)); ct = cos(th(:)); cp = cos(ph(:)); sp = sin(ph(:));
    dirs = [st.*cp, st.*sp, ct];    % N×3

    % Nearly boundary radius r0
    r0 = mask_radius * (1 - delta_boundary);

    % Evaluate V at ±r0·θ
    P   = r0 * dirs;
    Vp  = Fint(P(:,1),  P(:,2),  P(:,3));
    Vm  = Fint(-P(:,1), -P(:,2), -P(:,3));

    % Quadrature on S^2
    sdel = sum(w_ang(:) .* (Vp(:) .* Vm(:)));
end
