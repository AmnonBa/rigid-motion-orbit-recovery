function plan = build_bisys_sampling_plan(Lmax, R, Tabc, Gaunt, rows_per_c, seed, verbose)
% BUILD_BISYS_SAMPLING_PLAN
% ------------------------------------------------------------------------------
% Construct a *deterministic* sampling plan of tuples (l1, l2, a, b) for every
% (ℓ3, c) block used to assemble the degree-3 invariant linear systems (“Bisys”).
% The plan is reused across translations in the boundary-weighted (BW) pipeline,
% ensuring a one-to-one, row-wise comparison between ORACLE and BW assemblies.
%
% For each target block (ℓ3, c):
%   1) Enumerate all admissible harmonic degree pairs (l1, l2) that satisfy:
%        • triangle condition: |l1 − l2| ≤ ℓ3 ≤ l1 + l2
%        • parity condition:   l1 + l2 + ℓ3 is even
%        • nonempty Gaunt block Gaunt{l1,l2→ℓ3}
%   2) Sample `rows_per_c` rows:
%        • Choose (l1, l2) uniformly from the admissible set.
%        • Choose shell indices (a, b) by importance sampling proportional to
%          |Tabc(a,b,c)|, falling back to uniform if all weights are zero.
%   3) Record the row as (l1, l2, a, b).
%
% No row normalization is performed here; this function only returns the *plan*.
%
% INPUTS
%   Lmax        : maximum spherical-harmonic degree (non-negative integer).
%   R           : number of radial shells (positive integer).
%   Tabc        : triple radial overlap tensor, size R×R×R (T(a,b,c)).
%   Gaunt       : cell array Gaunt{l1+1, l2+1, ℓ3+1} of size
%                 (2ℓ3+1) × ((2l1+1)(2l2+1)); empty/zero ⇒ inadmissible pair.
%   rows_per_c  : target number of rows per (ℓ3, c) block (positive integer).
%   seed        : RNG seed for reproducibility (default: 0).
%   verbose     : if true/1 prints per-ℓ3 summary (default: 1).
%
% OUTPUT
%   plan        : cell array of size (Lmax+1) × R.
%                 plan{ℓ3+1, c} is an N×4 integer matrix whose rows are:
%                   [l1, l2, a, b], with N ≤ rows_per_c (if pairs are scarce).
%
% NOTES
%   • If no admissible (l1,l2) pairs exist for a certain ℓ3, the corresponding
%     plan entries will be empty (0×4).
%   • If |Tabc(:,:,c)| sums to zero, we sample (a,b) uniformly over {1..R}².
%   • Setting the same `seed` guarantees identical plans across runs.
% ------------------------------------------------------------------------------

% ------------------------------ Defaults & checks ------------------------------
if nargin < 6 || isempty(seed),    seed    = 0; end
if nargin < 7 || isempty(verbose), verbose = 1; end
assert(Lmax >= 0 && floor(Lmax) == Lmax,          'Lmax must be a non-negative integer.');
assert(R    >= 1 && floor(R)    == R,             'R must be a positive integer.');
assert(rows_per_c >= 1 && floor(rows_per_c) == rows_per_c, 'rows_per_c must be a positive integer.');
assert(iscell(Gaunt) && numel(Gaunt) >= (Lmax+1)^3, 'Gaunt must be a (Lmax+1)^3 cell array.');
assert(isequal(size(Tabc), [R R R]),               'Tabc must be R×R×R.');

rng(seed);  % deterministic plan

% Precompute (2ℓ+1) per ℓ (kept for documentation/clarity; not used directly)
% (Retained to mirror related code paths and help readers.)
%mL = arrayfun(@(l) 2*l+1, 0:Lmax); 

plan = cell(Lmax+1, R);

% --------------------------------- Main loop ----------------------------------
for ell3 = 0:Lmax

    % ---- Step 1: enumerate admissible (l1,l2) pairs for this ℓ3 ----
    pairs = [];  % each row: [l1, l2]
    for l1 = 0:Lmax
        for l2 = 0:Lmax
            % Triangle + parity conditions
            if ~(abs(l1 - l2) <= ell3 && ell3 <= l1 + l2)
                continue; 
            end
            if bitand(l1 + l2 + ell3, 1) ~= 0
                continue; 
            end

            % Nonempty Gaunt block required
            Gmap = Gaunt{l1+1, l2+1, ell3+1};
            if isempty(Gmap) || ~any(Gmap(:))
                continue; 
            end

            pairs = [pairs; l1, l2];
        end
    end
    if isempty(pairs)
        pairs = zeros(0, 2);  % explicit empty
    end

    % ---- Step 2: build a plan for each shell index c ----
    for c = 1:R
        % Importance weights over (a,b) via |Tabc(a,b,c)|
        M  = abs(Tabc(:,:,c));           % R×R (non-negative)
        p  = M(:);
        ps = sum(p);

        if ps <= 0
            % No information in Tabc slice ⇒ fallback to uniform over {1..R}²
            p  = ones(R*R, 1);
            ps = numel(p);
        end
        p = p / ps;  % probability vector (sums to 1)

        % Choose up to rows_per_c candidate rows
        rows = zeros(rows_per_c, 4, 'int32');
        k = 1;

        while k <= rows_per_c
            if isempty(pairs)
                % No admissible (l1,l2) ⇒ cannot add more rows
                break;
            end

            % (l1,l2): uniform over admissible pairs
            id = randi(size(pairs,1));
            l1 = pairs(id,1);
            l2 = pairs(id,2);

            % (a,b): importance sampling by |Tabc(a,b,c)|
            ab = randsample(R*R, 1, true, p);
            [a, b] = ind2sub([R, R], ab);

            % (Optional) re-check Gaunt block (defensive)
            Gmap = Gaunt{l1+1, l2+1, ell3+1};
            if isempty(Gmap) || ~any(Gmap(:))
                continue;  % skip degenerate rows
            end

            rows(k,:) = [l1, l2, a, b];
            k = k + 1;
        end

        % If we could not fill all rows (e.g., no pairs), trim zeros
        rows = rows(1:max(0, k-1), :);

        % Save plan for (ℓ3, c)
        plan{ell3+1, c} = rows;
    end

    % ---- Optional: progress printout ----
    if verbose
        fprintf('  [PLAN][ℓ=%d] rows per c=%d | admissible (l1,l2)=%d\n', ...
            ell3, rows_per_c, size(pairs,1));
    end
end
end
