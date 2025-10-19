%% ================== Helper: one trial with ratio-of-sums (BATCHED, accumulative) ==================
function [M2_avg, M3_avg, M2_seq, M3_seq] = run_one_trial_ratio_of_sums( ...
    f, cfg, target, M, sigma, varargin)
% RUN_ONE_TRIAL_RATIO_OF_SUMS  Batched Monte Carlo, ratio-of-sums, accumulative.
%   - Accepts single clean image f (S×S).
%   - Generates up to M noisy replicates in batches (optional antithetic).
%   - Calls se2_to_so2_M2M3_single_v3 on an S×S×B stack each iteration.
%   - Returns final ratio-of-sums AND the prefix sequences for k = 1..M.
%
% Name-Value:
%   'BatchSize'  (default 128)
%   'Antithetic' (default true)
%
% Notes:
%   - Outputs are complex ratios; take real(...) outside if desired.
%   - M2_seq(k) = (sum_{i=1..k} M2_val_i) / (sum_{i=1..k} D_i).
%   - Same for M3_seq.

% ---- parse options ----
p = inputParser;
p.addParameter('BatchSize', 128, @(x)isnumeric(x) && isscalar(x) && x>=1);
p.addParameter('Antithetic', true, @(x)islogical(x) && isscalar(x));
p.parse(varargin{:});
BATCH    = p.Results.BatchSize;
USE_ANTI = p.Results.Antithetic;

% ---- accumulators ----
num_sum_M2 = 0.0;            % complex numerator accumulator
num_sum_M3 = 0.0;            % complex numerator accumulator
D_sum      = 0.0;            % real denominator accumulator

% ---- prefix arrays (complex; real(...) can be applied by caller) ----
M2_seq = nan(1, M);
M3_seq = nan(1, M);

% ---- bookkeeping ----
S = size(f,1);
remaining = M;
f = double(f);
k0 = 0;                      % how many observations have been accumulated so far
tiny = 1e-30;                % sign-preserving zero guard (for prefixes)

while remaining > 0
    if USE_ANTI
        % generate up to 'take' with antithetic pairing; safe for odd 'remaining'
        pairs   = max(1, floor(BATCH/2));
        take    = min(remaining, 2*pairs);
        usePair = ceil(take/2);                          % <-- important: ceil to handle odd tails
        N    = randn(S,S,usePair) * sigma;
        Fpos = f + N;                                    % S x S x usePair
        Fneg = f - N;                                    % S x S x usePair
        Fbatch = cat(3, Fpos, Fneg);                     % S x S x (2*usePair)
        if size(Fbatch,3) > take                         % trim down to exactly 'take'
            Fbatch = Fbatch(:,:,1:take);
        end
    else
        take   = min(remaining, BATCH);
        N      = randn(S,S,take) * sigma;
        Fbatch = f + N;                                  % S x S x take
    end

    % ---- evaluate batch ----
    outB = se2_to_so2_M2M3_single_batched(Fbatch, cfg, target);
    m2_b = reshape(outB.M2_val, 1, []);                  % [1 x ntake], complex
    m3_b = reshape(outB.M3_val, 1, []);                  % [1 x ntake], complex
    d_b  = reshape(outB.D_scalar, 1, []);                % [1 x ntake], real
    ntake = numel(d_b);

    % ---- prefix update for this batch using cumulative sums (no per-sample loop) ----
    cs_m2 = cumsum(m2_b);                                % [1 x ntake]
    cs_m3 = cumsum(m3_b);
    cs_d  = cumsum(d_b);

    idx          = (k0+1) : (k0+ntake);                  % positions in global sequences
    denom_prefix = D_sum + cs_d;                         % [1 x ntake]
    ok           = abs(denom_prefix) > tiny;             % guard tiny denominators

    % fill only where denom is safe
    M2_seq(idx(ok)) = (num_sum_M2 + cs_m2(ok)) ./ denom_prefix(ok);
    M3_seq(idx(ok)) = (num_sum_M3 + cs_m3(ok)) ./ denom_prefix(ok);
    % (entries with ~ok remain NaN; caller can ignore or back-fill)

    % ---- accumulate totals and advance ----
    num_sum_M2 = num_sum_M2 + cs_m2(end);
    num_sum_M3 = num_sum_M3 + cs_m3(end);
    D_sum      = D_sum      + cs_d(end);

    remaining = remaining - ntake;
    k0 = k0 + ntake;
end

% ---- finalize (handle empty or tiny denominator gracefully) ----
if abs(D_sum) <= tiny
    M2_avg = NaN; M3_avg = NaN;
else
    M2_avg = num_sum_M2 / D_sum;
    M3_avg = num_sum_M3 / D_sum;
end
end
