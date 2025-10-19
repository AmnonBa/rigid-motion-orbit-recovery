function Bisys = assemble_bisys_oracle_from_plan(Ftrue, Gaunt, Tabc, plan, row_normalize, Lmax, R)
% ASSEMBLE_BISYS_ORACLE_FROM_PLAN
% ------------------------------------------------------------------------------
% Build the degree-3 invariant linear systems (“Bisys”) per (ℓ,c) using a
% PREDEFINED sampling plan of rows. This is useful for a fair, row-wise
% comparison against boundary-weighted (BW) invariants where the SAME plan
% is used to assemble both oracle and BW systems.
%
% Row model (oracle, for each listed tuple (l1,l2,a,b) in `plan{ℓ+1,c}`):
%   alpha = Gaunt{l1,l2→ℓ} * vec(Ftrue_{l1}(:,a) * Ftrue_{l2}(:,b)^T)   (m3×1)
%   tau   = Tabc(a,b,c)                                                 (scalar)
%   row   = (tau * alpha)'                                              (1×m3)
%   rhs   =  tau * (alpha' * s3),   s3 = Ftrue_{ℓ}(:,c)                 (scalar)
%
% If `row_normalize` is true, each row is scaled by its L2 norm to improve
% numerical conditioning: row /= ||row|| and rhs /= ||row||.
%
% INPUTS
%   Ftrue         : cell {Lmax+1,1}, Ftrue{ℓ+1} is (2ℓ+1)×R spherical-harmonic
%                   coefficients for degree ℓ across R shells.
%   Gaunt         : cell Gaunt{l1+1,l2+1,ℓ+1} of size (2ℓ+1)×((2l1+1)(2l2+1)).
%   Tabc          : triple radial overlap tensor, size R×R×R, entry T(a,b,c).
%   plan          : cell {Lmax+1,R}. plan{ℓ+1,c} is an N×4 array whose rows are
%                   [l1 l2 a b] tuples to instantiate oracle rows.
%   row_normalize : logical flag; if true, normalize rows & RHS by row L2 norm.
%   Lmax          : maximum degree (non-negative integer).
%   R             : number of shells (positive integer).
%
% OUTPUT
%   Bisys         : (Lmax+1)×R struct with fields:
%                     .A  → N×(2ℓ+1)  (assembled rows for degree ℓ and shell c)
%                     .b  → N×1       (assembled RHS)
%
% NOTES
%   • This function does NOT randomize anything; it strictly follows `plan`.
%   • If a selected (l1,l2) has an empty/zero Gaunt block, or produces a
%     degenerate row, the row will be left as zeros (after optional normalization).
%   • Use this to compare, one-for-one, oracle rows vs BW rows built from the
%     same sampling plan.
% ------------------------------------------------------------------------------

% --------------------------- basic checks ---------------------------
assert(iscell(Ftrue) && numel(Ftrue)==Lmax+1, 'Ftrue must be a cell of length Lmax+1.');
for ell = 0:Lmax
    F_ell = Ftrue{ell+1};
    assert(ismatrix(F_ell) && size(F_ell,1)==(2*ell+1) && size(F_ell,2)==R, ...
        'Ftrue{%d} must be (2*ell+1)×R for ell=%d.', ell+1, ell);
end
assert(iscell(plan) && all(size(plan)==[Lmax+1, R]), ...
    'plan must be a cell array of size (Lmax+1)×R.');
assert(islogical(row_normalize) || isnumeric(row_normalize), ...
    'row_normalize must be logical.');

% --------------------------- main assembly --------------------------
Bisys = repmat(struct('A',[],'b',[]), Lmax+1, R);

for ell3 = 0:Lmax
    m3 = 2*ell3 + 1;

    for c = 1:R
        rows = plan{ell3+1, c};                     % N×4, each row: [l1 l2 a b]
        if isempty(rows)
            Bisys(ell3+1,c).A = zeros(0, m3);
            Bisys(ell3+1,c).b = zeros(0, 1);
            continue;
        end

        N = size(rows,1);
        A = zeros(N, m3);
        b = zeros(N, 1);

        s3 = Ftrue{ell3+1}(:, c);                   % (2ℓ3+1)×1

        for n = 1:N
            l1 = rows(n,1);  l2  = rows(n,2);
            a  = rows(n,3);  bsh = rows(n,4);

            % Gaunt block and coefficient vectors
            Gmap = Gaunt{l1+1, l2+1, ell3+1};       % (m3)×((2l1+1)(2l2+1))
            if isempty(Gmap) || ~any(Gmap(:))
                % leave zero row
                continue;
            end

            f1 = Ftrue{l1+1}(:, a);
            f2 = Ftrue{l2+1}(:, bsh);
            if ~any(f1) || ~any(f2)
                % leave zero row
                continue;
            end

            v12   = reshape(f1 * (f2.'), [], 1);    % vec outer product
            alpha = Gmap * v12;                      % (m3×1)

            if ~all(isfinite(alpha)) || norm(alpha) < 1e-16
                % leave zero row
                continue;
            end

            tau  = Tabc(a, bsh, c);                 % scalar
            rowv = (tau * alpha).';                  % (1×m3)
            rhs  =  tau * (alpha.' * s3);            % scalar

            A(n,:) = rowv;
            b(n)   = rhs;
        end

        % Optional L2 row-normalization (recommended for stability)
        if row_normalize && ~isempty(A)
            rn = sqrt(sum(abs(A).^2, 2));           % N×1
            z  = rn > 0;
            A(z,:) = A(z,:) ./ rn(z);
            b(z)   = b(z)   ./ rn(z);
        end

        Bisys(ell3+1,c).A = A;
        Bisys(ell3+1,c).b = b;
    end
end
end
