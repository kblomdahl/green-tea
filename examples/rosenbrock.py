#!/usr/bin/env python3

import sys
import yaml

params = yaml.safe_load(sys.stdin)
if not params:
    exit(1)

a, b = 1, 100
x, y = float(params['x']), float(params['y'])

print((a - x)**2 + b * (y - x*x)**2)
