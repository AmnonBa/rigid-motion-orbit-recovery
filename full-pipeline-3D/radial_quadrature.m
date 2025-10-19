function [r_nodes, w_nodes] = radial_quadrature(nr)
% RADIAL_QUADRATURE  Trapezoidal-rule nodes and weights on [0, 1]
% ------------------------------------------------------------------------------
% Construct a simple, uniform radial quadrature suitable for integrals of the
% form ∫_0^1 f(r) dr. The grid starts slightly away from 0 to avoid the
% singularity that can appear in downstream formulas (e.g., when combined with
% r^2 factors). Endpoints are weighted with half-weights (standard trapezoid).
%
% INPUT
%   nr        - Number of radial nodes (integer ≥ 2 recommended)
%
% OUTPUTS
%   r_nodes   - [nr x 1] monotonically increasing nodes in (0, 1]
%               (starts at 1e-4 to avoid r=0)
%   w_nodes   - [nr x 1] trapezoidal weights such that
%                   ∫_0^1 f(r) dr  ≈  sum_n f(r_nodes(n)) * w_nodes(n)
%
% NOTES
%   • If your integral uses the spherical measure r^2 dr, multiply the
%     returned weights by r_nodes.^2 when forming sums, e.g.:
%         rw = (r_nodes.^2) .* w_nodes;
%   • The tiny offset (1e-4) prevents evaluating functions exactly at r=0,
%     which is helpful when later combining with terms like 1/r or when
%     building r^2-weighted inner products.
%
% EXAMPLE
%   nr = 512;
%   [r, w] = radial_quadrature(nr);
%   % Approximate ∫_0^1 r^2 dr = 1/3:
%   approx = sum( (r.^2) .* w );   % ≈ 1/3
% ------------------------------------------------------------------------------

    % Uniform grid on (1e-4, 1]
    r_nodes = linspace(1e-4, 1, nr).';   % column vector

    % Trapezoidal weights on the uniform grid
    dr = r_nodes(2) - r_nodes(1);
    w_nodes = dr * ones(size(r_nodes));
    w_nodes([1, end]) = dr / 2;          % half-weights at the ends
end
