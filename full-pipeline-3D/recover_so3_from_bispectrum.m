function [Fhat, diagOut] = recover_so3_from_bispectrum(invData, opts)
% RECOVER_SO3_FROM_BISPECTRUM
% -------------------------------------------------------------------------
% Constructive recovery in SO(3) from degree-2 (Gram) and degree-3 (bispectrum)
% invariants using frequency marching. This implements the linear-in-f^ell
% step per ell, assembling many equations across shell pairs.
%
% INPUT:
%   invData :
%     .Lmax            - maximum degree
%     .R               - number of shells
%     .mL              - vector [m_0,...,m_Lmax], where m_l = 2*l+1
%     .Gaunt           - cell Gaunt{l1+1,l2+1,l3+1} of size (m3 x (m1*m2)),
%                        mapping vec(f^{l1}⊗f^{l2}) -> coeffs in H_{l3}
%                        (computed by numerical quadrature; complex-valued)
%     .G2              - cell G2{l+1}: R x R Gram := (F^{l})^H * F^{l}
%     .Bisys           - struct array with linear systems for each (ell,c):
%                        For each ell=0..Lmax, for each shell c=1..R:
%                          Bisys(ell+1,c).A : (#eq x m_ell)  (complex)
%                          Bisys(ell+1,c).b : (#eq x 1)      (complex)
%                        where rows come from many (l1,l2) & shell-pairings.
%     .F_l1_known      - (OPTIONAL) true coefficients at ell=1 to fix gauge
%                        (m1 x R). If absent, we will pick an arbitrary basis.
%
%   opts :
%     .verbose         - 0/1/2 (default 1). 2 = very chatty.
%     .reg_lambda      - Tikhonov (>=0), default 1e-8
%     .min_eq_per_col  - minimum #equations per unknown to accept solve (default 3)
%     .condition_warn  - threshold to warn on cond(A) (default 1e8)
%
% OUTPUT:
%   Fhat  : cell of recovered coefficients per ell, size (m_l x R)
%   diagOut : struct with diagnostics:
%       .perEll(shell) condition numbers, residual norms, #eqs, ranks...
%
% -------------------------------------------------------------------------

if ~isfield(opts,'verbose'),        opts.verbose = 1; end
if ~isfield(opts,'reg_lambda'),     opts.reg_lambda = 1e-8; end
if ~isfield(opts,'min_eq_per_col'), opts.min_eq_per_col = 3; end
if ~isfield(opts,'condition_warn'), opts.condition_warn = 1e8; end

Lmax   = invData.Lmax;
R      = invData.R;
mL     = invData.mL;

Fhat    = cell(Lmax+1,1);
diagOut = struct();
diagOut.perEll = cell(Lmax+1,1);

% ---------- ℓ = 0 ----------
ell = 0; m = mL(ell+1);
if opts.verbose
    fprintf('[RECOVER] ℓ=%d (m=%d): solving shell-wise linear systems from Bisys.\n', ell, m);
end

% Prepare a Gram-based fallback for ℓ=0 (rank-1)
f0_gram = [];
if ~isempty(invData.G2) && numel(invData.G2) >= 1 && ~isempty(invData.G2{1})
    G0 = invData.G2{1};
    G0 = (G0 + G0')/2;                            % ensure Hermitian
    [V,D] = eig(G0,'vector');
    [lam,ix] = max(real(D));
    if isfinite(lam) && lam > 0 && ~isempty(ix)
        v = V(:,ix);
        f0_gram = (sqrt(lam) * (v.')).';          % column form (R×1), we'll index per-shell
        % We store column for convenience; each shell takes its scalar
    end
end

any_rows_l0 = false;
for c = 1:R
    if ~isempty(invData.Bisys(ell+1,c).A) && size(invData.Bisys(ell+1,c).A,1)>0
        any_rows_l0 = true; break;
    end
end

Fhat{ell+1} = zeros(m,R);  % m==1
for c = 1:R
    A = invData.Bisys(ell+1,c).A;   % (#eq x 1)
    b = invData.Bisys(ell+1,c).b;

    if isempty(A) || size(A,1)==0
        % --- Fallback: take value from Gram rank-1 if available
        if ~isempty(f0_gram)
            x = f0_gram(c);
        else
            x = 0;
        end
        Fhat{ell+1}(:,c) = x;
        cn = NaN; rn = 0; nr = 0; k = 0; neq = 0;
    else
        % Regularized LS: (A^H A + λI)x = A^H b
        x = solve_reg_ls(A, b, opts.reg_lambda);
        Fhat{ell+1}(:,c) = x;
        [cn, rn, nr, k] = diag_ls(A, b, x);
        neq = size(A,1);
    end

    diagOut.perEll{ell+1}(c).condA = cn;
    diagOut.perEll{ell+1}(c).res2  = rn;
    diagOut.perEll{ell+1}(c).normb = nr;
    diagOut.perEll{ell+1}(c).rankA = k;
    diagOut.perEll{ell+1}(c).neq   = neq;

    if opts.verbose>1 && ~isempty(A)
        fprintf('  [ℓ=%d,c=%d] eq=%d, rank=%d, cond(A)=%.2e, ||Ax-b||/||b||=%.2e\n', ...
                ell,c, size(A,1), k, cn, rn/max(nr,eps));
    end
    if ~isempty(A) && cn > opts.condition_warn && opts.verbose
        fprintf('  [WARN] ℓ=%d,c=%d ill-conditioned: cond(A)=%.2e\n', ell,c,cn);
    end
end

% ---------- ℓ = 1 (gauge) ----------
if Lmax >= 1
    ell = 1; m = mL(ell+1);
    if isfield(invData,'F_l1_known') && ~isempty(invData.F_l1_known)
        if opts.verbose
            fprintf('[RECOVER] ℓ=1: using provided gauge (oracle ℓ=1) of size %dx%d.\n', m, R);
        end
        Fhat{ell+1} = invData.F_l1_known;
    else
        if opts.verbose
            fprintf('[RECOVER] ℓ=1: no oracle given. Picking an arbitrary representative via LS.\n');
        end
        tmp = zeros(m,R);
        for c = 1:R
            A = invData.Bisys(ell+1,c).A;
            b = invData.Bisys(ell+1,c).b;
            if isempty(A) || size(A,1) < opts.min_eq_per_col*m
                if opts.verbose
                    fprintf('  [ℓ=1,c=%d] too few eq (%d) vs unknowns (%d). Zeroing.\n', ...
                            c, size(A,1), m);
                end
                x = zeros(m,1);
            else
                x = solve_reg_ls(A, b, opts.reg_lambda);
            end
            tmp(:,c) = x;
            [cn, rn, nr, k] = diag_ls(A,b,x);
            diagOut.perEll{ell+1}(c).condA = cn;
            diagOut.perEll{ell+1}(c).res2  = rn;
            diagOut.perEll{ell+1}(c).normb = nr;
            diagOut.perEll{ell+1}(c).rankA = k;
            diagOut.perEll{ell+1}(c).neq   = size(A,1);
            if opts.verbose>1 && ~isempty(A)
                fprintf('  [ℓ=%d,c=%d] eq=%d, rank=%d, cond(A)=%.2e, ||Ax-b||/||b||=%.2e\n', ...
                        ell,c, size(A,1), k, cn, rn/max(nr,eps));
            end
        end
        Fhat{ell+1} = tmp;
    end
end

% ---------- March ℓ = 2..Lmax ----------
for ell = 2:Lmax
    m = mL(ell+1);
    if opts.verbose
        fprintf('[RECOVER] ℓ=%d (m=%d): assemble LS per shell from Bisys.\n', ell, m);
    end
    X = zeros(m,R);
    for c = 1:R
        A = invData.Bisys(ell+1,c).A;   % rows collected from (l1,l2) pairs and shell-pairs
        b = invData.Bisys(ell+1,c).b;

        if isempty(A) || size(A,1)==0
            if opts.verbose
                fprintf('  [ℓ=%d,c=%d] no equations. Zeroing.\n', ell,c);
            end
            x = zeros(m,1);
        else
            if size(A,1) < opts.min_eq_per_col*m && opts.verbose
                fprintf('  [ℓ=%d,c=%d] few eq (%d) vs unknowns (%d). Using reg LS anyway.\n', ...
                        ell,c, size(A,1), m);
            end
            x = solve_reg_ls(A, b, opts.reg_lambda);
        end

        X(:,c) = x;

        [cn, rn, nr, k] = diag_ls(A, b, x);
        diagOut.perEll{ell+1}(c).condA = cn;
        diagOut.perEll{ell+1}(c).res2  = rn;
        diagOut.perEll{ell+1}(c).normb = nr;
        diagOut.perEll{ell+1}(c).rankA = k;
        diagOut.perEll{ell+1}(c).neq   = size(A,1);

        if opts.verbose>1 && ~isempty(A)
            fprintf('  [ℓ=%d,c=%d] eq=%d, rank=%d, cond(A)=%.2e, ||Ax-b||/||b||=%.2e\n', ...
                    ell,c, size(A,1), k, cn, rn/max(nr,eps));
        end
        if ~isempty(A) && cn > opts.condition_warn && opts.verbose
            fprintf('  [WARN] ℓ=%d,c=%d ill-conditioned: cond(A)=%.2e\n', ell,c,cn);
        end
    end
    Fhat{ell+1} = X;
end

% ---------- Optional: check G2 consistency (post) ----------
if opts.verbose
    fprintf('[POST] Checking degree-2 Gram consistency errors per ℓ (||F^H F - G2||_F / ||G2||_F).\n');
end
diagOut.G2_rel_err = zeros(Lmax+1,1);
for ell = 0:Lmax
    if isempty(invData.G2{ell+1}), continue; end
    G2t = invData.G2{ell+1};
    F   = Fhat{ell+1};
    if isempty(F), diagOut.G2_rel_err(ell+1) = NaN; continue; end
    G2h = F' * F;
    rel = norm(G2h - G2t,'fro') / max(norm(G2t,'fro'), eps);
    diagOut.G2_rel_err(ell+1) = rel;
    if opts.verbose
        fprintf('  [ℓ=%d] Gram rel.err = %.3e\n', ell, rel);
    end
end
end

% ---------- helpers ----------
function x = solve_reg_ls(A,b,lambda)
if isempty(A)
    x = zeros(0,1);
    return;
end
% Solve (A^H A + λI)x = A^H b
[~, n] = size(A);
x = (A'*A + lambda*eye(n)) \ (A' * b);
end

function [condA, res2, normb, rnk] = diag_ls(A,b,x)
if isempty(A)
    condA = NaN; res2 = 0; normb = norm(b); rnk = 0; return;
end
s = svd(A,'econ');
if isempty(s), condA = NaN; else, condA = s(1)/max(s(end),eps); end
res2  = norm(A*x - b);
normb = norm(b);
rnk   = sum(s > max(size(A))*eps*max(s));
end
