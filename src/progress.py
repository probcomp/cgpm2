# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import sys
import time

def report_progress(percentage, stream):
    progress = ' ' * 30
    fill = int(percentage * len(progress))
    progress = '[' + '=' * fill + progress[fill:] + ']'
    stream.write('\r{} {:1.2f}%'.format(progress, 100 * percentage))
    stream.flush()

def proportion_done(N, S, iters, start):
    if S is None:
        p_seconds = 0
    else:
        p_seconds = (time.time() - start) / S
    if N is None:
        p_iters = 0
    else:
        p_iters = float(iters)/N
    return max(p_iters, p_seconds)

def transition_generic(kernels, N=None, S=None, progress=None):
    if N is None and S is None:
        N = 1
    if progress is None:
        progress = True
    iters = 0
    start = time.time()
    while True and kernels:
        for kernel in kernels:
            p = proportion_done(N, S, iters, start)
            if progress:
                report_progress(p, sys.stdout)
            if p >= 1.:
                break
            kernel()
        else:
            iters += 1
            continue
        break
