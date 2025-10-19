function idx = fft_index(m, N)
%FFT_INDEX  Map integer Fourier mode m to MATLAB's 1-based DFT bin index.
%   idx = FFT_INDEX(m, N) returns the 1-based index in 1..N corresponding to
%   the integer frequency mode m for an N-point DFT under MATLAB's binning.
%
%   DEFINITION
%     MATLAB stores DFT samples in a 1-based vector Y(1:N). The bin indices
%     correspond to integer modes via
%         idx = mod(m, N) + 1 .
%     Hence:
%       m = 0      -> idx = 1      (DC)
%       m = 1..N-1 -> idx = m+1
%       m = -1     -> idx = N
%       m = -k     -> idx = N-k+1  (wrap-around)
%
%   INPUTS
%     m : Integer mode (scalar or array). Can be negative or exceed N; wraps
%         modulo N. Non-integers will be reduced modulo N as well, but the
%         intended use is with integer modes.
%     N : Positive integer, the DFT length.
%
%   OUTPUT
%     idx : Same shape as m, with values in 1..N (double).
%
%   EXAMPLES
%     fft_index(0, 8)    % -> 1    (DC)
%     fft_index(1, 8)    % -> 2
%     fft_index(-1, 8)   % -> 8
%     fft_index(9, 8)    % -> 2    (since 9 ≡ 1 mod 8)
%     fft_index([-2 0 3], 8)  % -> [7 1 4]
%
%   SEE ALSO: FFT, IFFT
%
%   Notes:
%     - This function is vectorized: m may be a scalar or an array.
%     - No input validation is enforced beyond the formula; call ASSERT or
%       VALIDATEATTRIBUTES externally if needed.

idx = mod(m, N) + 1;

end
