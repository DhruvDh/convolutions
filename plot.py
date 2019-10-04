def perf(k, n, m):
    flops = ((2 * k * k) - 1) * (n - k + 1) * (m - k + 1)
    flops = flops * 10e-9

    time_taken = flops / 1638.4

    pixels_produced = (n - k + 1) * (m - k + 1)

    pixels_per_second = pixels_produced / time_taken
    return pixels_per_second * 10e-9

import matplotlib.pyplot as plt
import pandas as pd

kernels = [3, 5, 7, 9, 11, 13, 15]

plt.figure()
plt.plot(kernels, [perf(k, 1024, 768) for k in kernels])
plt.savefig('plot.png')