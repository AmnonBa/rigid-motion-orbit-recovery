function Y = sphharmY(l, m, theta, phi)
% SPHHARMY  Physics-normalized spherical harmonics Y_l^m(θ, φ), orthonormal on S^2
% ------------------------------------------------------------------------------
%   Y = sphharmY(l, m, theta, phi)
%
% Compute the complex spherical harmonic Y_l^m(θ, φ) using the physics
% (Condon–Shortley) normalization, i.e., the set {Y_l^m} is orthonormal on
% the unit sphere with respect to the surface measure dΩ = sinθ dθ dφ:
%
%       ∫_{S^2} Y_l^m(θ,φ) * conj(Y_{l'}^{m'}(θ,φ)) dΩ = δ_{l l'} δ_{m m'} .
%
% INPUTS
%   l       - nonnegative integer degree (l = 0,1,2,...)
%   m       - integer order with |m| ≤ l
%   theta   - polar angle(s) θ in [0, π], vector or array of any shape
%   phi     - azimuth angle(s) φ in [0, 2π), same size as theta
%
% OUTPUT
%   Y       - row vector (1 × N) of complex values for Y_l^m evaluated at the
%             N points (theta(:), phi(:)). Output is always returned as a row.
%
% NORMALIZATION & CONVENTIONS
%   • Associated Legendre polynomials are produced by MATLAB's 'legendre' with
%     'unnorm' (unnormalized) option. We then apply the physics normalization:
%
%         N_lm = sqrt( (2l+1)/(4π) * (l - |m|)! / (l + |m|)! )
%
%     and the Condon–Shortley phase (−1)^m is applied for negative m via the
%     standard relationship Y_l^{-m} = (−1)^m conj(Y_l^{m}).
%
% VECTORIZATION
%   • theta and phi can be vectors or arrays (same shape). The function flattens
%     them internally and returns a 1×N row vector of values.
%
% EXAMPLES
%   % Grid on the sphere:
%   [th, ph] = ndgrid(linspace(0,pi,64), linspace(0,2*pi,128));
%   Y20 = sphharmY(2, 0, th, ph);   % (1 × (64*128))
%
%   % Check orthonormality numerically on a crude grid (not a true quadrature):
%   Y21 = sphharmY(2, 1, th, ph);
%   dΩ  = sin(th(:)) * (th(2,1)-th(1,1)) * (ph(1,2)-ph(1,1));
%   approx_inner = Y20 * conj(Y21.').*dΩ;  % ~ 0 (not exact on uniform grid)
%
% NUMERICAL NOTES
%   • Uses 'legendre(l, cosθ, ''unnorm'')' which returns a (l+1)×N array
%     containing P_l^m for m=0..l. We then select the |m| row and build Y_l^m.
%   • A tiny EPS guard is used when forming factorial ratios to avoid div/0.
%   • For performance, the result is returned as a row vector. Use reshape if
%     you want the original grid shape: reshape(Y, size(theta)).
% ------------------------------------------------------------------------------

    % ---- Input checks (lightweight) -------------------------------------
    if ~(isscalar(l) && isnumeric(l) && isfinite(l) && l>=0 && l==floor(l))
        error('l must be a nonnegative integer.');
    end
    if ~(isscalar(m) && isnumeric(m) && isfinite(m) && abs(m) <= l && m==floor(m))
        error('|m| must be <= l and m must be an integer.');
    end
    if ~isequal(size(theta), size(phi))
        error('theta and phi must have the same size.');
    end

    % ---- Prepare angular inputs -----------------------------------------
    ct = cos(theta(:).');         % 1 × N  (row vector)
    N  = numel(theta);

    % ---- Unnormalized associated Legendre functions ---------------------
    % Punn is (l+1) × N with rows m=0..l
    Punn = legendre(l, ct, 'unnorm');   % double precision
    mm   = abs(m);
    Plm  = squeeze(Punn(mm+1, :));      % 1 × N

    % ---- Physics normalization ------------------------------------------
    % N_l = sqrt((2l+1)/(4π)), then include the (l±m)! ratio
    Nl = sqrt((2*l + 1) / (4*pi));
    Nm = Nl * sqrt( factorial(l-mm) / max(factorial(l+mm), eps) );

    % ---- Assemble Y_l^m --------------------------------------------------
    if m >= 0
        % Positive (or zero) m: Y_l^m(θ,φ) = N_{lm} P_l^m(cosθ) e^{i m φ}
        Y = Nm * Plm .* exp(1i * m * phi(:).');
    else
        % Negative m: Y_l^{-m} = (−1)^m * conj(Y_l^{|m|})
        Ypos = Nm * Plm .* exp(1i * mm * phi(:).');  % Y_l^{|m|}
        Y    = (-1)^m * conj(Ypos);
    end

    % Always return as row (1 × N)
    Y = Y(:).';
end
