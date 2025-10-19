function [Acell, bcell] = build_for_shell_c(c, F, Gaunt, Tabc, admissible, Lmax, R, mL, rows_per_c)
% BUILD_FOR_SHELL_C
% ------------------------------------------------------------------------------
% Assemble the Bispectrum (degree-3) linear system rows, *for a single radial
% shell c*, across all spherical-harmonic degrees ℓ = 0..Lmax.
%
% This is the per-shell worker used by the fast, model-consistent M^3 → Bisys
% “sketch”. For each target (ℓ,c) block we randomly draw admissible pairs
% (l1,l2) and random shell indices (a,b), form
%
%   alpha = G_{l1,l2→ℓ} * vec( F_{l1}(:,a) * F_{l2}(:,b)^T )     (m3×1)
%   s3    = F_ℓ(:,c)                                            (m3×1)
%   tau   = T(a,b,c)                                            (scalar)
%
% and append one normalized row:
%
%   row = (tau * alpha)'      ∈ R^{1×m3}
%   rhs =  tau * (alpha' s3)  ∈ R
%
% Rows are L2-normalized (row / ||row||, rhs / ||row||) for numerical stability.
%
% INPUTS
%   c           : (1×1) target shell index (1..R) for which to build rows.
%   F           : cell(Lmax+1,1). F{ℓ+1} is (2ℓ+1)×R SH coefficients (columns = shells).
%   Gaunt       : cell(Lmax+1,Lmax+1,Lmax+1), Gaunt{l1+1,l2+1,ℓ+1} maps vec outer products
%                 into the ℓ block (size (2ℓ+1) × ((2l1+1)(2l2+1))).
%   Tabc        : R×R×R triple radial overlap tensor T(a,b,c).
%   admissible  : cell(Lmax+1,1), admissible{ℓ+1} is K×2 list of (l1,l2) pairs
%                 that satisfy triangle/parity constraints for target ℓ.
%   Lmax        : (scalar) maximum SH degree.
%   R           : (scalar) number of radial shells.
%   mL          : (Lmax+1)×1 vector with mL(ℓ+1) = 2ℓ+1 (precomputed).
%   rows_per_c  : (scalar) requested number of rows per (ℓ,c) block.
%
% OUTPUTS
%   Acell       : cell(Lmax+1,1), Acell{ℓ+1} is (#rows × (2ℓ+1)) matrix of rows.
%   bcell       : cell(Lmax+1,1), bcell{ℓ+1} is (#rows × 1) vector of RHS values.
%
% GUARANTEES & BEHAVIOR
%   • If an (ℓ,c) has no admissible (l1,l2) pairs or fails to assemble any
%     valid rows, Acell{ℓ+1} and bcell{ℓ+1} are empty (0×m3, 0×1).
%   • The routine makes up to max(10*rows_per_c, 20000) trials to reach
%     rows_per_c valid rows per (ℓ,c); it stops early if not enough can be found.
%   • If more than rows_per_c valid rows are accumulated (rare), it trims by
%     random subsampling to exactly rows_per_c rows.
%
% SEE ALSO
%   estimate_Bisys_from_M3_sketch_fast, build_bisys_sampling_plan
% ------------------------------------------------------------------------------

    % Preallocate output containers (one block per ℓ)
    Acell = cell(Lmax+1, 1);
    bcell = cell(Lmax+1, 1);

    % Loop over target degrees ℓ
    for ell = 0:Lmax
        m3 = mL(ell+1);
        pairs = admissible{ell+1};  % admissible (l1,l2) for this ℓ

        % If no admissible pairs, return empty block
        if isempty(pairs)
            Acell{ell+1} = zeros(0, m3);
            bcell{ell+1} = zeros(0, 1);
            continue;
        end

        % Target shell column s3 for this (ℓ,c)
        s3 = F{ell+1}(:, c);        % (2ℓ+1)×1

        % Growable accumulators for rows
        A_rows = zeros(0, m3);
        b_rows = zeros(0, 1);

        % Randomized sketch: keep sampling until we have enough valid rows
        ns = 0;
        maxTrials = max(10*rows_per_c, 20000);
        while (ns < rows_per_c) && (maxTrials > 0)
            maxTrials = maxTrials - 1;

            % Random admissible (l1,l2); random shell indices (a,b)
            id = randi(size(pairs,1));
            l1 = pairs(id,1);
            l2 = pairs(id,2);
            a  = randi(R);
            b  = randi(R);

            % Fetch participating columns
            f1 = F{l1+1}(:, a);
            f2 = F{l2+1}(:, b);
            if ~any(f1) || ~any(f2), continue; end

            % Gaunt map for (l1,l2) → ℓ
            Gmap = Gaunt{l1+1, l2+1, ell+1};
            if isempty(Gmap) || ~any(Gmap(:)), continue; end

            % alpha = G * vec(f1 * f2^T)
            v12   = reshape(f1 * (f2.'), [], 1);   % ((2l1+1)(2l2+1))×1
            alpha = Gmap * v12;                    % (2ℓ+1)×1
            if ~any(alpha), continue; end

            % Radial triple overlap
            tau = Tabc(a, b, c);                   % scalar

            % One unnormalized row and RHS
            row = (tau * alpha).';                 % 1×(2ℓ+1)
            rhs =  tau * (alpha.' * s3);           % 1×1

            % Row L2-normalization check (reject ill-conditioned)
            rn = norm(row);
            if rn < 1e-14 || ~isfinite(rn) || ~isfinite(rhs), continue; end

            % Store normalized row
            A_rows(end+1, :) = row / rn;           %#ok<AGROW>
            b_rows(end+1, 1) = rhs / rn;           %#ok<AGROW>
            ns = ns + 1;
        end

        % If we overshot (rare), trim back to rows_per_c uniformly at random
        if size(A_rows,1) > rows_per_c
            idx = randperm(size(A_rows,1), rows_per_c);
            A_rows = A_rows(idx, :);
            b_rows = b_rows(idx, :);
        end

        % Finalize this (ℓ,c) block
        Acell{ell+1} = A_rows;
        bcell{ell+1} = b_rows;
    end
end
