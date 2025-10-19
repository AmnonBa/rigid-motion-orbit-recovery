function Gaunt = build_gaunt_tables_w3j(Lmax)
% BUILD_GAUNT_TABLES_W3J
% ------------------------------------------------------------------------------
% Construct Gaunt coefficient lookup tables up to degree Lmax using closed-form
% Wigner–3j symbols (computed stably via Racah's formula in log-space).
%
% The Gaunt tensor maps an outer product of spherical-harmonic coefficient
% blocks at degrees (l1, l2) into the block at degree l3:
%
%   alpha_{l3} = G_{l1,l2→l3} * vec( F_{l1} * F_{l2}^T ),      size(alpha_{l3}) = (2l3+1)×1
%
% where:
%   • G_{l1,l2→l3} ∈ ℝ^{(2l3+1) × ((2l1+1)(2l2+1))}
%   • vec(F_{l1} * F_{l2}^T) stacks the (2l1+1)×(2l2+1) outer product column-wise.
%
% OUTPUT
%   Gaunt : cell(Lmax+1, Lmax+1, Lmax+1)
%           Gaunt{l1+1, l2+1, l3+1} has size (2*l3+1) × ((2*l1+1)*(2*l2+1)).
%           If (l1,l2,l3) is inadmissible (triangle/parity constraints), the
%           entry is a zero matrix of the appropriate size.
%
% INPUT
%   Lmax  : maximum spherical-harmonic degree (non-negative integer)
%
% ADMISSIBILITY (selection rules)
%   • Triangle: |l1 − l2| ≤ l3 ≤ l1 + l2
%   • Parity:   l1 + l2 + l3 is even
%
% NUMERICAL NOTES
%   • The geometric/normalization factor includes the (0,0,0) Wigner–3j term:
%       C = sqrt((2l1+1)(2l2+1)(2l3+1)/(4π)) * wigner3j(l1,l2,l3, 0,0,0)
%   • Individual entries use (-1)^{m3} C * wigner3j(l1,l2,l3, m1, m2, -m3)
%   • Wigner–3j is evaluated with a log-domain Racah sum for stability.
%
% COMPLEXITY
%   Roughly O(Lmax^4) coefficients; for moderate Lmax (≤ 10–20) this is practical.
%
% SEE ALSO
%   wigner3j (local), triangle_ok (local)
% ------------------------------------------------------------------------------

    Gaunt = cell(Lmax+1, Lmax+1, Lmax+1);

    for l1 = 0:Lmax
        m1s = -l1:l1; n1 = numel(m1s);
        for l2 = 0:Lmax
            m2s = -l2:l2; n2 = numel(m2s);

            % Precompute column indices to place (m1,m2) into vec-outer layout
            col_idx = reshape( (0:n1*n2-1)+1, n1, n2 );

            for l3 = 0:Lmax
                % Admissibility: triangle + even parity
                if ~( abs(l1-l2) <= l3 && l3 <= l1+l2 && mod(l1+l2+l3,2) == 0 )
                    Gaunt{l1+1,l2+1,l3+1} = zeros(2*l3+1, n1*n2);
                    continue;
                end

                m3s = -l3:l3; n3 = numel(m3s);
                G   = zeros(n3, n1*n2);

                % Overall normalization factor for m=(0,0,0)
                C = sqrt((2*l1+1)*(2*l2+1)*(2*l3+1)/(4*pi)) * wigner3j(l1,l2,l3, 0,0,0);

                if C == 0
                    Gaunt{l1+1,l2+1,l3+1} = G;
                    continue;
                end

                % Fill each m3 row by summing over admissible (m1,m2) with m1+m2=m3
                for i3 = 1:n3
                    m3  = m3s(i3);
                    row = zeros(1, n1*n2);

                    for i1 = 1:n1
                        m1 = m1s(i1);
                        m2 = m3 - m1;             % selection rule m1 + m2 + m3 = 0

                        if m2 < -l2 || m2 >  l2
                            continue;
                        end

                        % Map (m1,m2) into column index of vec-outer product
                        j2  = m2 - (-l2) + 1;
                        col = col_idx(i1, j2);

                        % Entry uses (-1)^{m3} C * (l1 l2 l3; m1 m2 -m3)
                        val = (-1)^m3 * C * wigner3j(l1,l2,l3, m1, m2, -m3);
                        row(col) = val;
                    end

                    G(i3,:) = row;
                end

                Gaunt{l1+1,l2+1,l3+1} = G;
            end
        end
    end
end


% --- Stable Wigner–3j via Racah formula in log-space ---------------------------
function val = wigner3j(l1,l2,l3,m1,m2,m3)
% WIGNER3J  Stable evaluation of the Wigner–3j symbol
%   ( l1  l2  l3 )
%   ( m1  m2  m3 )
%
% Uses Racah's summation in log-domain to mitigate overflow/underflow.
% Returns zero when selection rules are violated.

    % Selection rules
    if (m1 + m2 + m3) ~= 0, val = 0; return; end
    if any([abs(m1) > l1, abs(m2) > l2, abs(m3) > l3]), val = 0; return; end
    if (l1 < 0) || (l2 < 0) || (l3 < 0), val = 0; return; end
    if ~triangle_ok(l1,l2,l3), val = 0; return; end

    % Phase
    phase = (-1)^(l1 - l2 - m3);

    % Triangle/factorial block (in log)
    t1 = l1 + l2 - l3;  t2 = l1 - l2 + l3;  t3 = -l1 + l2 + l3;  t4 = l1 + l2 + l3 + 1;
    logTri = 0.5 * ( gammaln(t1+1) + gammaln(t2+1) + gammaln(t3+1) - gammaln(t4+1) );

    % Magnetic factorials (in log)
    logM = 0.5 * ( ...
        gammaln(l1-m1+1) + gammaln(l1+m1+1) + ...
        gammaln(l2-m2+1) + gammaln(l2+m2+1) + ...
        gammaln(l3-m3+1) + gammaln(l3+m3+1) );

    % Racah sum bounds
    kmin = max([0, l2-l3-m1, l1-l3+m2]);
    kmax = min([l1-m1, l2+m2, t1]);

    % Accumulate Racah sum in linear domain (terms individually stable)
    S = 0;
    for k = kmin:kmax
        a1 = l1 - m1 - k;     a2 = l2 + m2 - k;     a3 = t1 - k;
        b1 = k;               b2 = l3 - l2 + m1 + k; b3 = l3 - l1 - m2 + k;

        logTerm = -( ...
            gammaln(a1+1) + gammaln(a2+1) + gammaln(a3+1) + ...
            gammaln(b1+1) + gammaln(b2+1) + gammaln(b3+1) );

        S = S + (-1)^k * exp(logTerm);
    end

    % Final value
    val = phase * exp(logTri + logM) * S;
end


% --- Helper: triangle inequality on degrees ------------------------------------
function ok = triangle_ok(l1,l2,l3)
    ok = (abs(l1-l2) <= l3) && (l3 <= l1+l2);
end
