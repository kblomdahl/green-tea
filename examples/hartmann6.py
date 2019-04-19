#!/usr/bin/env python3

import sys
import yaml
import numpy as np
import math

params = yaml.safe_load(sys.stdin)
if not params:
    exit(1)

x = np.asarray([
    params['x1'],
    params['x2'],
    params['x3'],
    params['x4'],
    params['x5'],
    params['x6'],
], np.float32)

alpha = np.asarray([
    1.0,
    1.2,
    3.0,
    3.2
], np.float32)

A = np.asarray([
    [10, 3, 17, 3.5, 1.7, 8],
    [0.05, 10, 17, 0.1, 8, 14],
    [3, 3.5, 1.7, 10, 17, 8],
    [17, 8, 0.05, 10, 0.1, 14]
], np.float32)

P = np.asarray([
    [1312, 1696, 5569,  124, 8283, 5886],
    [2329, 4135, 8307, 3736, 1004, 9991],
    [2348, 1451, 3522, 2883, 3047, 6650],
    [4047, 8828, 8732, 5743, 1091,  381]
], np.float32) * 1e-4

result = 0.0

for i in range(4):
    inner = 0.0

    for j in range(6):
        inner += A[i, j] * (x[j] - P[i, j])**2

    result += alpha[i] * math.exp(-inner)

print(-result)
