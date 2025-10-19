function [rings_hat, ring_metrics] = invert_all_rings_nonunit(M2_stack_non, B_cells, Rings_ref, K_r, inv)
%INVERT_ALL_RINGS_NONUNIT  Per-radius (ring) inversion from M2/M3 on a circle.
%   [rings_hat, ring_metrics] = INVERT_ALL_RINGS_NONUNIT(M2_stack_non, B_cells, Rings_ref, K_r, inv)
%   reconstructs each angular ring profile (one per radius) from its
%   non-unitary 2nd-order autocorrelation in angle (M2) and a local
%   bispectrum block (M3 information) using a phase-recovery routine.
%
%   INPUTS
%     M2_stack_non : [Nphi x nr] real matrix. Column ir contains the φ-domain
%                    non-unitary M2 (i.e., inverse-DFT of |X|^2) for ring ir.
%     B_cells      : 1 x nr cell array. B_cells{ir} is a (2K+1)x(2K+1) complex
%                    bispectrum sub-block for modes m1,m2 ∈ {-K..K} on ring ir,
%                    arranged so that B(m1+K+1,m2+K+1) ≈ X[m1] X[m2] X[-m1-m2].
%     Rings_ref    : [nr x Nphi] reference ring signals (for diagnostics only).
%                    Not used for reconstruction (no “cheating”); only for metrics.
%     K_r          : [nr x 1] integer vector. Per-ring bandlimit K for modes ±K.
%     inv          : struct with fields:
%                      - Nphi (positive integer): angular grid size.
%                      - print_per_ring (logical): print per-ring debug lines.
%
%   OUTPUTS
%     rings_hat    : [nr x Nphi] reconstructed ring signals (rows are rings).
%     ring_metrics : [nr x 4] diagnostics vs bandlimited reference:
%                    [relL2, corr, SNRdB, eq_rot_deg]
%                      relL2     = ||x_hat_aligned - ref_K|| / ||ref_K||
%                      corr      = Pearson correlation with ref_K
%                      SNRdB     = 10*log10(var(ref_K)/var(error))
%                      eq_rot_deg= estimated rotation (degrees) aligning x_hat to ref
%
%   DETAILS
%     - Uses non-unitary FFT conventions (MATLAB’s default).
%     - For each ring, we compute |X| from M2, recover phases from normalized
%       bispectrum using a simple progressive (triangular) estimator with a fixed
%       gauge (θ0=0, θ1=0), then synthesize X and IFFT back to spatial angle.
%     - No reference information is used in the reconstruction; references are
%       only used to report metrics.
%
%   SEE ALSO: FFT, IFFT, BANDLIMIT_RING_NONUNIT, FFT_INDEX

Nphi = inv.Nphi;
nr   = size(M2_stack_non,2);
rings_hat    = zeros(nr, Nphi);
ring_metrics = zeros(nr,4); % [relL2, corr, SNRdB, eq_rot_deg] vs bandlimited ref (report-only)

for ir=1:nr
    K = K_r(ir);
    M2_phi = M2_stack_non(:,ir).';     % 1×Nphi (non-unitary φ-domain M2)
    Bblk   = B_cells{ir};

    % Reconstruction (no alignment to reference)
    xr_hat_raw = invert_ring_from_M2M3_nonunit(M2_phi, Bblk, Nphi, K); % row
    rings_hat(ir,:) = xr_hat_raw;

    % ---------------- Diagnostics only (no impact on reconstruction) -------------
    ref_full = Rings_ref(ir,:);
    ref_K    = bandlimit_ring_nonunit(ref_full, K, Nphi);
    [xr_hat_for_metrics, rot_deg] = align_ring_to_ref(ref_K, xr_hat_raw); % copy only

    relL2 = norm(xr_hat_for_metrics - ref_K) / max(norm(ref_K), eps);
    c     = corr(ref_K(:), xr_hat_for_metrics(:));
    snrdb = 10*log10( var(ref_K,1) / max(var(xr_hat_for_metrics - ref_K,1), 1e-20) );
    ring_metrics(ir,:) = [relL2, c, snrdb, rot_deg];

    % ---------------- Self-consistency checks on raw result ----------------------
    [relM2, relB] = ring_self_consistency_nonunit(xr_hat_raw, M2_phi, Bblk, Nphi, K);
    if isfield(inv,'print_per_ring') && inv.print_per_ring
        if ir<=24 || mod(ir,8)==0 || ir==nr
            fprintf('[ring %03d | K=%3d] relM2=%.3e  relB=%.3e\n', ir-1, K, relM2, relB);
        end
    end
end
end

function xr_hat = invert_ring_from_M2M3_nonunit(M2_phi, B_block, Nphi, K)
%INVERT_RING_FROM_M2M3_NONUNIT  Per-ring inversion (non-unitary): magnitude from M2, phase from M3.
%   xr_hat = INVERT_RING_FROM_M2M3_NONUNIT(M2_phi, B_block, Nphi, K) reconstructs
%   a single ring signal x(φ) from its non-unitary angular M2 (φ-domain) and a
%   local bispectrum block around modes |m|≤K. The method:
%     1) FFT(M2_phi) to obtain |X|^2 on the frequency grid; take sqrt → |X|.
%     2) Normalize the bispectrum to obtain complex phase constraints C(m1,m2).
%     3) Recover phases progressively with a fixed gauge θ0=0, θ1=0.
%     4) Synthesize X and IFFT back to obtain xr_hat (enforce real symmetry).
%
%   INPUTS
%     M2_phi : [1 x Nphi] real vector, non-unitary M2 in φ-domain (ifft(|X|^2)).
%     B_block: [(2K+1) x (2K+1)] complex sub-block of bispectrum samples for
%              m1,m2 ∈ {-K..K}, with rows/cols ordered by m+K+1.
%     Nphi   : DFT length (angular samples).
%     K      : Bandlimit (keep modes m=0, ±1, …, ±K).
%
%   OUTPUT
%     xr_hat : [1 x Nphi] real reconstructed ring (spatial angle domain).
%
%   NOTES
%     - Non-unitary FFT convention (MATLAB).
%     - The gauge (θ0=0, θ1=0) fixes the inherent global rotation/flip; later
%       synchronization can re-introduce the correct global rotation if needed.
%     - Weights W ∝ |X[m1] X[m2] X[-m1-m2]| stabilize the phase averaging.
%
%   SEE ALSO: FFT, IFFT, FFT_INDEX

% Magnitude from M2 (non-unitary)
M2_hat = real(fft(M2_phi, Nphi));       % = |X|^2
Aabs_full = sqrt(max(M2_hat, 0));       % |X|

% Gather |X_m| for |m|<=K
Aabs = zeros(2*K+1,1);
for m=-K:K, Aabs(m+K+1) = Aabs_full(fft_index(m,Nphi)); end

% Normalize bispectrum → pure phase constraints C
C = zeros(2*K+1, 2*K+1);
W = zeros(2*K+1, 2*K+1);
for m1=-K:K
    for m2=-K:K
        m3 = -(m1+m2);
        if abs(m3)<=K
            den = Aabs(m1+K+1)*Aabs(m2+K+1)*Aabs(m3+K+1);
            if den>1e-12
                C(m1+K+1, m2+K+1) = B_block(m1+K+1, m2+K+1) / den;
                W(m1+K+1, m2+K+1) = den;
            end
        end
    end
end

% Phase recovery with gauge θ0=0, θ1=0 (later synchronization can reinsert rotation)
theta = zeros(2*K+1,1); theta(K+1)=0; if K>=1, theta(K+2)=0; end
hasC = @(m1,m2) (abs(C(m1+K+1,m2+K+1))>0);
argC = @(m1,m2) angle(C(m1+K+1,m2+K+1));

% Progressive estimation for m=2..K using triangular relations
for m=2:K
    ests = []; wts = [];
    for j=1:m-1
        if hasC(j,m-j)
            ph  = argC(j,m-j);
            est = theta(K+1+j) + theta(K+1+(m-j)) - ph;
            ests(end+1) = wrap_to_pi(est); 
            wts(end+1)  = W(j+K+1, m-j+K+1); 
        end
    end
    if isempty(ests)
        theta(K+1+m) = 0;
    else
        theta(K+1+m) = angle(sum( wts(:)'.*exp(1i*ests(:)') ));
    end
end
% Enforce conjugate symmetry: θ(-m) = -θ(m)
for m=1:K, theta(K+1-m) = -theta(K+1+m); end

% Synthesize full spectrum X[k] and inverse FFT (non-unitary)
X_full = zeros(1,Nphi);
X_full(1) = Aabs(K+1)*exp(1i*theta(K+1)); % m=0
for m=1:K
    amp = Aabs(K+1+m); phs = theta(K+1+m);
    X_full(fft_index( m,Nphi)) = amp*exp( 1i*phs);
    X_full(fft_index(-m,Nphi)) = amp*exp(-1i*phs);
end
xr_hat = real(ifft(X_full,'symmetric'));
end

function [relM2, relB] = ring_self_consistency_nonunit(x_row, M2_phi_given, B_block_given, Nphi, K)
%RING_SELF_CONSISTENCY_NONUNIT  Self-consistency residuals (M2 and bispectrum).
%   [relM2, relB] = RING_SELF_CONSISTENCY_NONUNIT(x_row, M2_phi_given, B_block_given, Nphi, K)
%   computes relative errors between:
%     - The M2 predicted by x_row and the given M2 in φ-domain, and
%     - The bispectrum block predicted by x_row and the given bispectrum block.
%
%   INPUTS
%     x_row          : [1 x Nphi] reconstructed ring.
%     M2_phi_given   : [1 x Nphi] non-unitary φ-domain M2 (ifft of |X|^2).
%     B_block_given  : [(2K+1) x (2K+1)] complex block for modes |m1|,|m2|≤K.
%     Nphi, K        : angular length and bandlimit.
%
%   OUTPUTS
%     relM2 : Relative L2 error of M2 (φ-domain), normalized by ||M2_phi_given||.
%     relB  : Relative Frobenius error of bispectrum block, normalized by ||B_block_given||.
%
%   SEE ALSO: FFT, IFFT, FFT_INDEX

X = fft(x_row, Nphi);
M2_phi_est = real(ifft(abs(X).^2));              % non-unitary M2 in φ-domain
relM2 = norm(M2_phi_est - M2_phi_given)/max(norm(M2_phi_given),1e-20);

[m1,m2] = ndgrid(-K:K,-K:K);
Xm1 = X(fft_index(m1,Nphi));
Xm2 = X(fft_index(m2,Nphi));
Xm3 = X(fft_index(-(m1+m2),Nphi));
Bhat = Xm1.*Xm2.*Xm3;           % non-unitary bispectrum (no scaling)
B_est_block = reshape(Bhat, 2*K+1, 2*K+1);
relB = norm(B_est_block - B_block_given)/max(norm(B_block_given),1e-20);
end

function [x_aligned, rot_deg] = align_ring_to_ref(x_ref, x)
%ALIGN_RING_TO_REF  Align a ring signal to a reference by circular shift.
%   [x_aligned, rot_deg] = ALIGN_RING_TO_REF(x_ref, x) finds the cyclic shift
%   that maximizes circular cross-correlation between x and x_ref, returns the
%   shifted signal and the equivalent rotation in degrees.
%
%   INPUTS
%     x_ref : [1 x N] reference signal on the circle.
%     x     : [1 x N] signal to align.
%
%   OUTPUTS
%     x_aligned : [1 x N] circshift(x, -shift) maximizing corr with x_ref.
%     rot_deg   : Scalar rotation in degrees equivalent to the shift.
%
%   NOTES
%     - Uses FFT-based correlation via ifft(conj(FFT(x_ref)).*FFT(x)).
%     - Orientation: positive shift corresponds to rotation by +360*(shift/N) degrees.

N = numel(x);
X  = fft(x);
XR = fft(x_ref);
xc = ifft( conj(XR).*X );
[~,kmax] = max(real(xc));
shift = kmax-1;
x_aligned = circshift(x, [0, -shift]);
rot_deg = shift / N * 360;
end

function ang = wrap_to_pi(a)
%WRAP_TO_PI  Wrap angles (radians) to (-pi, pi].
%   ang = WRAP_TO_PI(a) returns angles equivalent to a modulo 2π,
%   mapped into the open-closed interval (-π, π].
ang = mod(a + pi, 2*pi) - pi;
end
