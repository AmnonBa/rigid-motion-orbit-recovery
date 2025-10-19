function yK = bandlimit_ring_nonunit(y_row, K, Nphi)
%BANDLIMIT_RING_NONUNIT  Low-pass filter Fourier modes on a 1D ring (non-unitary FFT).
%   yK = BANDLIMIT_RING_NONUNIT(y_row, K, Nphi) keeps the DC component and the
%   first K positive/negative angular Fourier modes of the input row y_row, 
%   zeroing all higher modes in the length-Nphi DFT, and returns the filtered
%   signal yK via an inverse FFT.
%
%   This is useful when bandlimiting angular samples on a ring (e.g., per-radius
%   slices in polar/spherical pipelines) before reconstruction or denoising.
%
%   INPUTS
%     y_row : [1 x N] or [N x 1] real or complex vector of angular samples.
%             If Nphi > numel(y_row), zero-padding is implied by FFT; if
%             Nphi < numel(y_row), the FFT treats y_row as length Nphi via
%             periodic truncation.
%     K     : Nonnegative integer; number of positive-frequency modes to keep.
%             The modes retained are m = 0, ±1, ±2, ..., ±K.
%     Nphi  : Positive integer; FFT length (number of angular samples in the
%             working grid). Typically equals numel(y_row).
%
%   OUTPUT
%     yK    : Filtered signal in the spatial/angle domain, same orientation as
%             y_row (MATLAB’s IFFT returns a column if input is column, etc.).
%             For real y_row and symmetric selection, yK is real (enforced via
%             IFFT(...,'symmetric')).
%
%   DETAILS
%     - The routine forms Y = FFT(y_row, Nphi), builds a logical mask that keeps
%       the DC bin and the bins corresponding to ±m for m = 1..K using an index
%       helper FFT_INDEX(m,Nphi), zeros the rest, and returns IFFT of masked Y.
%     - The FFT is MATLAB’s non-unitary convention (no 1/sqrt(N) factors).
%     - The helper FFT_INDEX must map integer mode m ∈ {-(Nphi-1),...,Nphi-1}
%       to the 1-based MATLAB DFT bin in [1..Nphi] with proper wrap-around.
%
%   ASSUMPTIONS / REQUIREMENTS
%     - K should satisfy 0 ≤ K ≤ floor((Nphi-1)/2) to avoid overlapping positive
%       and negative bins when Nphi is even. Larger K will just keep all modes.
%     - Function FFT_INDEX(m,Nphi) must exist on the path.
%
%   EXAMPLE
%     % Keep only DC and first 3 harmonics of a noisy angular profile:
%     Nphi  = 256;
%     theta = linspace(0, 2*pi, Nphi+1); theta(end) = [];
%     y     = cos(2*theta) + 0.5*sin(3*theta) + 0.1*randn(size(theta));
%     y3    = bandlimit_ring_nonunit(y, 3, Nphi);   % keeps m=0,±1,±2,±3
%
%   SEE ALSO: FFT, IFFT
%
%   Author: <your name>, <affiliation> (<year>)

Y = fft(y_row, Nphi);

% Build keep-mask for modes m = 0, ±1, ..., ±K
keep = false(1, Nphi);
keep(1) = true; % DC term (m = 0)

for m = 1:K
    keep(fft_index( m, Nphi)) = true;
    keep(fft_index(-m, Nphi)) = true;
end

% Zero out higher frequencies
Y(~keep) = 0;

% Inverse transform; enforce real output when spectrum is Hermitian
yK = real(ifft(Y, 'symmetric'));

end
