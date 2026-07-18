import numpy as np
import math

# TODO document!

# Function that computes FPT counts vector probability
def log_FPT_prob(vec, a0=1):
    L = math.log(len(vec), 2)
    bb_m = 0.5 # beta-binomial mean
    bb_s = 2*a0 # TODO ??
    log_prob = 0
    tmp = vec

    for l in reversed(range(L)):
        ind_odd = range(1, 2**(l-1), 2)
    # TODO complete 1st

# Function that computes the logarithm of the multinomial coefficient
def log_multi_coef(vec):
    s = sum(vec)
    coef = math.lgamma(s + 1)
    for i in vec:
        coef -= math.lgamma(i + 1)
    return coef # TODO is there a way to be exact?


def FPT_log_likelihood(vec, a0=1):
    pass
    m = sum(vec)
    L = math.log(len(vec), 2)

    log_lik = L * m * math.log(2) - log_multi_coef(vec) # TODO complete 2nd



def dir_mult_log_likelihood(vec, a0=1):
    m = sum(vec)
    L = math.log(len(vec), 2)
    # TODO complete 3rd


def simulate():

    L = 10
    m = 200
    N = 10

    lst = ([1] * 16 + [2] * 16) * (2 ** 5)
    s = sum(lst)
    p1_vec = [i / s for i in lst]
    p0_vec = [1 / (2 ** L)] * (2 ** L)

    FPT_1 = [None] * N
    FPT_0_1 = [None] * N
    FPT_10 = [None] * N

    dir_mult = [None] * N
    chisq_mult = [None] * N
    LR = [None] * N
    correct_lik = [None] * N

    for i in range(N):
        null_n_vec = np.random.multinomial(m, p0_vec)
        FPT_1[i] = FPT_log_likelihood(null_n_vec,a0=1)
        FPT_0_1[i] = FPT_log_likelihood(null_n_vec,a0=0.1)
        FPT_10[i] = FPT_log_likelihood(null_n_vec,a0=10)

        alt_n_vec = np.random.multinomial(m, p1_vec)
        print(n_vec)

    print(FPT_1)

    print(len(p1_vec))
    print(p1_vec)
    print(len(p0_vec))
    print(p0_vec)

simulate()
