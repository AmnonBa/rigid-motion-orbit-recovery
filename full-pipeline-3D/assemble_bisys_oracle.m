function [Bisys, admissible] = assemble_bisys_oracle(Ftrue, Gaunt, Tabc, Lmax, R, eq_per_c)
% ASSEMBLE_BISYS_ORACLE
% ------------------------------------------------------------------------------
% Build the degree-3 invariant linear systems ("Bisys") per (ell, c) from the
% *oracle* spherical-harmonic coefficients Ftrue, the Gaunt tables, and the
% triple radial overlap tensor Tabc.
%
% For each harmonic degree ell=0..Lmax and shell index c=1..R, this function
% assembles a normalized linear system
%
%       Bisys(ell+1,c).A * s3 = Bisys(ell+1,c).b
%
% whose *row model* matches the oracle construction:
%   alpha = Gaunt{l1,l2->ell} * vec(Ftrue_{l1}(:,a) * Ftrue_{l2}(:,b)^T)     (m3×1)
%   tau   = Tabc(a,b,c)                                                      (scalar)
%   row   = (tau * alpha)'                                                   (1×m3)
%   rhs   =  tau * (alpha' * s3),  s3 = Ftrue_{ell}(:,c)                     (scalar)
%
% Each row is L2-normalized for numerical stability: row /= ||row||, rhs /= ||row||.
%
% INPUTS
%   Ftrue   : cell array, size {Lmax+1,1}. Ftrue{ell+1} is (2*ell+1)×R matrix
%             of SH coefficients at degree ell across R shells.
%   Gaunt   : cell array, Gaunt{l1+1,l2+1,ell+1} is (2*ell+1)×((2*l1+1)*(2*l2+1))
%             block that maps vec(F_l1 * F_l2^T) to degree-ell coupling.
%   Tabc    : 3-D tensor of size R×R×R with entries T(a,b,c) (triple radial overlaps).
%   Lmax    : maximum SH degree (non-negative integer).
%   R       : number of radial shells.
%   eq_per_c: target number of equations (rows) per (ell,c) block (positive integer).
%
% OUTPUTS
%   Bisys      : (Lmax+1)×R struct array with fields:
%                  .A  → (#rows)×(2*ell+1)   normalized left-hand rows
%                  .b  → (#rows)×1           normalized right-hand side
%   admissible : {Lmax+1,1} cell. admissible{ell+1} is a K×2 list of (l1,l2)
%                pairs that satisfy triangle parity constraints for degree ell.
%
% NOTES
%   • (l1,l2,ell) must satisfy the triangular inequality and parity (l1+l2+ell even).
%   • Row sampling:
%       - (l1,l2) is drawn uniformly from admissible pairs for degree ell.
%       - Column pair (a,b) is drawn from a categorical distribution ∝ |Tabc(a,b,c)|.
%         If Tabc(:,:,c) is all zeros, a diagonally-biased fallback is used.
%   • Robustness: rows with non-finite alpha, near-zero norm, or degenerate rhs are skipped.
%   • Printing: each (ell,c) reports how many rows were assembled (may be < eq_per_c).
%
% COMPLEXITY (rough order):
%   O( sum_{ell,c}  eq_per_c × [cost(Gaunt * vec(·))] )
%
% ------------------------------------------------------------------------------
% Amnon B. / 2025 — cleaned & documented version of the original routine.
% ------------------------------------------------------------------------------

% ------------------------- sanity checks -------------------------
assert(isscalar(Lmax) && Lmax>=0 && Lmax==floor(Lmax), 'Lmax must be a non-negative integer.');
assert(isscalar(R)    && R>=1    && R==floor(R),       'R must be a positive integer.');
assert(isscalar(eq_per_c) && eq_per_c>=0 && eq_per_c==floor(eq_per_c), ...
       'eq_per_c must be a non-negative integer.');

% Validate Ftrue layout
assert(iscell(Ftrue) && numel(Ftrue) == Lmax+1, 'Ftrue must be a cell array of length Lmax+1.');
for ell = 0:Lmax
    F_ell = Ftrue{ell+1};
    assert(ismatrix(F_ell) && size(F_ell,1) == 2*ell+1 && size(F_ell,2) == R, ...
          'Ftrue{%d} must be (2*ell+1)×R for ell=%d.', ell+1, ell);
end

% ----------------- collect admissible (l1,l2) per ell -----------------
admissible = cell(Lmax+1,1);
for ell = 0:Lmax
    pairs = [];
    for l1 = 0:Lmax
        for l2 = 0:Lmax
            if ~triangle_ok(l1, l2, ell)
                continue; 
            end
            if mod(l1 + l2 + ell, 2) ~= 0 
                continue; 
            end
            pairs = [pairs; l1, l2]; 
        end
    end
    admissible{ell+1} = pairs;
end

% ----------------- build column (a,b) samplers per c ------------------
% Probability ∝ |Tabc(a,b,c)|; fallback uses a diagonal-friendly kernel.
samplers = cell(R,1);
for c = 1:R
    M  = abs(Tabc(:,:,c));
    p  = M(:);
    ps = sum(p);
    if ps <= 0
        % Fallback: emphasize near-diagonal without being singular
        M = eye(R);
        for k = 1:R-1
            M = M + 0.5*(diag(ones(R-1,1), k) + diag(ones(R-1,1), -k));
        end
        p  = M(:);
        ps = sum(p);
    end
    samplers{c}.cdf = cumsum(p / ps);
end

% --------------- assemble Bisys rows per (ell,c) block ----------------
Bisys = repmat(struct('A', [], 'b', []), Lmax+1, R);

for ell = 0:Lmax
    pairs = admissible{ell+1};              % admissible (l1,l2) for this ell
    m3    = 2*ell + 1;

    for c = 1:R
        if isempty(pairs) || eq_per_c == 0
            Bisys(ell+1, c).A = zeros(0, m3);
            Bisys(ell+1, c).b = zeros(0, 1);
            fprintf('  [INV][ℓ=%d,c=%d] rows=0 (target=%d) [LOW]\n', ell, c, eq_per_c);
            continue;
        end

        A_rows  = zeros(0, m3);
        b_rows  = zeros(0, 1);
        nsamp   = 0;
        maxTrials = max(10*eq_per_c, 20000); % cap trials to avoid infinite loops

        while nsamp < eq_per_c && maxTrials > 0
            maxTrials = maxTrials - 1;

            % --- sample (l1,l2) uniformly from admissible set
            id  = randi(size(pairs,1));
            l1  = pairs(id,1);
            l2  = pairs(id,2);

            % --- sample (a,b) from |Tabc(:,:,c)|
            u   = rand;
            k   = find(samplers{c}.cdf >= u, 1, 'first');
            if isempty(k), k = numel(samplers{c}.cdf); end
            [a, b] = ind2sub([R, R], k);

            % --- build row and rhs from oracle model
            Gmap = Gaunt{l1+1, l2+1, ell+1};             % (m3)×((2*l1+1)*(2*l2+1))
            if isempty(Gmap) || ~any(Gmap(:)), continue; end

            f1 = Ftrue{l1+1}(:, a);
            f2 = Ftrue{l2+1}(:, b);
            if ~any(f1) || ~any(f2), continue; end

            v12   = reshape(f1 * (f2.'), [], 1);         % vec outer-product
            alpha = Gmap * v12;                           % (m3 × 1)
            if ~all(isfinite(alpha)) || norm(alpha) < 1e-16, continue; end

            tau = Tabc(a, b, c);                         % scalar radial coupling

            row = (tau * alpha).';                        % (1 × m3)
            s3  = Ftrue{ell+1}(:, c);                     % (m3 × 1)
            rhs =  tau * (alpha.' * s3);                  % scalar

            % --- L2 normalization of the row (and rhs coherently)
            rn = norm(row);
            if rn < 1e-16 || ~isfinite(rn) || ~isfinite(rhs), continue; end

            A_rows(end+1, :) = row / rn;                  
            b_rows(end+1, 1) = rhs / rn;                  
            nsamp = nsamp + 1;
        end

        Bisys(ell+1, c).A = A_rows;
        Bisys(ell+1, c).b = b_rows;

        fprintf('  [INV][ℓ=%d,c=%d] rows=%d (target=%d)%s\n', ...
            ell, c, size(A_rows,1), eq_per_c, tern(size(A_rows,1) < eq_per_c, ' [LOW]', ''));
    end
end
end

% =========================== Helpers (local functions) ===========================
function ok = triangle_ok(l1, l2, l3)
% Return true iff (l1,l2,l3) satisfies triangular inequality.
ok = (abs(l1 - l2) <= l3) && (l3 <= l1 + l2);
end

function t = tern(cond, a, b)
% Ternary helper: t = a if cond, else b.
if cond
    t = a;
else
    t = b;
end
end
