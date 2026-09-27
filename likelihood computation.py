import random
import time
import pandas as pd

import numpy as np
from scipy.special import gammaln, logsumexp
from collections import defaultdict
import itertools


def log_beta_binomial(k, n, a0):
    """Computes log Beta-Binomial pmf for left split k out of n."""
    if n == 0:
        return 0.0
    # Binomial coefficient + Beta ratio
    log_comb = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    log_beta_num = gammaln(k + a0) + gammaln(n - k + a0) - gammaln(n + 2 * a0)
    log_beta_den = 2 * gammaln(a0) - gammaln(2 * a0)
    return log_comb + log_beta_num - log_beta_den


def precompute_bits(U, max_depth):
    """Stage 1: Precompute binary bits for each observation and dimension."""
    m, P = U.shape
    # Scale coordinates to integer bit representation
    scaled = np.clip((U * (2 ** max_depth)).astype(np.int64), 0, (2 ** max_depth) - 1)
    bits = np.zeros((m, P, max_depth), dtype=np.int64)
    for d in range(max_depth):
        bits[:, :, d] = (scaled >> (max_depth - 1 - d)) & 1
    return bits


def compute_marginal_likelihood_DP(U, L, a0, quotas=None, pi=None):
    m, P = U.shape
    if quotas is None:
        quotas = tuple([L] * P)
    if pi is None:
        pi = {p: 1.0 / P for p in range(P)}

    bits = precompute_bits(U, L)

    # Memoization tables: store log_M and N indexed by (k_tuple, i_tuple)
    log_M = {}
    N_counts = {}

    # Helper to get all non-negative integer vectors summing to l capped by quotas
    def get_compositions(l):
        for comp in itertools.product(*[range(min(l, quotas[p]) + 1) for p in range(P)]):
            if sum(comp) == l:
                yield comp

    # Stage 2: Level L leaf counts
    for k in get_compositions(L):
        # Calculate leaf addresses for each observation
        addresses = np.zeros(m, dtype=np.int64)
        for p in range(P):
            if k[p] > 0:
                for b in range(k[p]):
                    addresses = (addresses << 1) | bits[:, p, b]

        # Count frequency per populated address
        unique_addrs, counts = np.unique(addresses, return_counts=True)
        for addr, count in zip(unique_addrs, counts):
            state = (k, (int(addr),))
            log_M[state] = 0.0  # Base case: log M = 0 at depth L
            N_counts[state] = int(count)

    # Stage 3: Bottom-up dynamic programming (l = L-1 down to 0)
    for l in range(L - 1, -1, -1):
        for k in get_compositions(l):
            # Identify valid candidate splits
            valid_splits = [p for p in range(P) if k[p] < quotas[p]]

            # Find all active addresses at level l
            active_addresses = set()
            for p in valid_splits:
                k_child = list(k)
                k_child[p] += 1
                k_child = tuple(k_child)
                for (state_k, state_i) in log_M.keys():
                    if state_k == k_child:
                        # Parent address is shifted right by 1
                        parent_addr = state_i[0] >> 1
                        active_addresses.add((parent_addr,))

            if l == 0:
                active_addresses = {(0,)}

            for i in active_addresses:
                branch_scores = []
                state = (k, i)

                for p in valid_splits:
                    k_child = list(k)
                    k_child[p] += 1
                    k_child = tuple(k_child)

                    i_low = (i[0] << 1,)
                    i_high = ((i[0] << 1) | 1,)

                    state_low = (k_child, i_low)
                    state_high = (k_child, i_high)

                    n_low = N_counts.get(state_low, 0)
                    n_high = N_counts.get(state_high, 0)
                    n_total = n_low + n_high

                    if n_total > 0:
                        N_counts[state] = n_total

                    log_M_low = log_M.get(state_low, 0.0)
                    log_M_high = log_M.get(state_high, 0.0)

                    log_beta_bin = log_beta_binomial(n_low, n_total, a0)
                    score = np.log(pi[p]) + log_beta_bin + log_M_low + log_M_high
                    branch_scores.append(score)

                if branch_scores:
                    log_M[state] = logsumexp(branch_scores)
                else:
                    log_M[state] = 0.0

    # Stage 4: Root Value
    root_state = (tuple([0] * P), (0,))
    log_fH1 = log_M.get(root_state, 0.0)
    return log_fH1
