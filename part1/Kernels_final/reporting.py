import matplotlib.pyplot as plt
import numpy as np

# Kernel labels
kernels = ["K1","K2","K3","K4","K5","K6","K7","K8","K9","K10","K11","Ultimate"]

# MFLOPS values (leave TBD as np.nan or 0)
regular = [
    384270.918,    # K1
    518144.379,    # K2
    1336248.057,   # K3
    1440315.598,   # K4
    np.nan,        # K5
    2598812.320,   # K6
    2819478.727,   # K7
    np.nan,        # K8
    589104.255,    # K9
    3903167.450,   # K10
    3273480.602,   # K11
    6448287.170    # Ultimate
]

optimized = [
    546772.120,    # K1
    619885.765,    # K2
    2010612.685,   # K3
    1408267.944,   # K4
    np.nan,        # K5
    3549230.859,   # K6
    3898264.612,   # K7
    np.nan,        # K8
    3891389.772,   # K9
    3890903.860,   # K10
    3240072.575,   # K11
    6512742.541    # Ultimate
]

x = np.arange(len(kernels))
width = 0.35

fig, ax = plt.subplots(figsize=(12,6))
rects1 = ax.bar(x - width/2, regular, width, label='Regular')
rects2 = ax.bar(x + width/2, optimized, width, label='Optimized')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MFLOPS')
ax.set_title('Matrix Multiplication Performance per Kernel')
ax.set_xticks(x)
ax.set_xticklabels(kernels)
ax.legend()

# Optional: show MFLOPS values on top of bars
for i in range(len(kernels)):
    if not np.isnan(regular[i]):
        ax.text(x[i]-width/2, regular[i]+50000, f'{int(regular[i]/1000)}k', ha='center', va='bottom', fontsize=8)
    if not np.isnan(optimized[i]):
        ax.text(x[i]+width/2, optimized[i]+50000, f'{int(optimized[i]/1000)}k', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()