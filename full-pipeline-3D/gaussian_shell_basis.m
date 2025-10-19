function [W, cvec] = gaussian_shell_basis(r_nodes, r_shells, sigma, w_nodes)
% GAUSSIAN_SHELL_BASIS
% ------------------------------------------------------------------------------
% Build a bank of radially Gaussian “shell” basis functions on [0, 1] and
% L2-normalize each shell with respect to the weighted inner product
%           ⟨f,g⟩ = ∫_0^1 f(r) g(r) r^2 dr
% using a supplied quadrature rule {r_nodes, w_nodes}.
%
% This is typically used to collapse continuous radial profiles into R
% smoothly localized shells, each centered at r_shells(s). The normalization
% makes each shell have unit energy under the discrete approximation of
% the r^2 dr inner product (i.e., per-shell energy equals 1).
%
% INPUTS
%   r_nodes   : [nr x 1] quadrature nodes on [0,1] (monotone increasing)
%   r_shells  : [S  x 1] shell centers (in [0,1]); S = number of shells
%   sigma     : scalar Gaussian bandwidth (same units as r) controlling shell
%               thickness. Smaller → thinner shells. A floor of 1e-6 is used
%               for numerical stability.
%   w_nodes   : [nr x 1] quadrature weights for ∫_0^1 • dr (e.g., trapezoid);
%               the r^2 factor is applied internally.
%
% OUTPUTS
%   W     : [S x nr] matrix of normalized shell basis evaluations:
%               W(s, n) = cvec(s) * exp(-0.5 * ((r_nodes(n) - r_shells(s))/sigma)^2)
%           so that each row s satisfies
%               sum_n W(s,n)^2 * (r_nodes(n)^2 * w_nodes(n)) ≈ 1
%   cvec  : [S x 1] per-shell normalization constants used to scale rows of G
%
% NUMERICAL NOTES
%   • Normalization uses the discrete r^2 dr weight: rw(n) = r_nodes(n)^2 * w_nodes(n).
%   • To avoid division by zero when sigma is extremely small, we clamp
%     sigma with max(sigma, 1e-6).
%   • A small epsilon (1e-14) is added under the square root to guard against
%     round-off when a shell is extremely narrow or poorly sampled.
%
% SHAPE CONVENTIONS
%   r_nodes, w_nodes, r_shells are treated as column vectors; W has one row
%   per shell and one column per radial node.
%
% EXAMPLE
%   nr       = 512;
%   r_nodes  = linspace(1e-4, 1, nr).';
%   w_nodes  = repmat((r_nodes(2)-r_nodes(1)), nr, 1); w_nodes([1,end]) = w_nodes(1)/2;
%   r_shells = linspace(0.1, 0.9, 8).';
%   sigma    = 0.04;
%   [W, c]   = gaussian_shell_basis(r_nodes, r_shells, sigma, w_nodes);
%   % Verify unit-energy rows under r^2 dr:
%   rw       = (r_nodes.^2).*w_nodes;
%   energy   = sum(W.^2 .* rw.', 2)    % ≈ ones(S,1)
% ------------------------------------------------------------------------------

    % Number of shells and nodes
    S  = numel(r_shells);
    nr = numel(r_nodes);

    % Evaluate unnormalized Gaussian bumps at all (shell, node) pairs.
    % Result G is S×nr with G(s,n) = exp(-0.5*((r(n)-r_s)/sigma)^2).
    sig = max(sigma, 1e-6);
    G   = exp(-0.5 * ((r_nodes(:).' - r_shells(:))./sig).^2);  % [S x nr]

    % Discrete r^2 dr weights at nodes
    rw = (r_nodes.^2) .* w_nodes;                               % [nr x 1]

    % Compute per-shell normalization so that each row has unit L2(r^2 dr) norm:
    %   c(s) = 1 / sqrt( sum_n G(s,n)^2 * rw(n) )
    denom = sum(G.^2 .* rw.', 2);                                % [S x 1]
    cvec  = 1 ./ sqrt(denom + 1e-14);

    % Apply normalization row-wise
    W = (cvec .* ones(1, nr)) .* G;                              % [S x nr]
end
