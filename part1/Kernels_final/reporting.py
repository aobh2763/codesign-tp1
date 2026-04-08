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
    764732.300,    # K5
    2598812.320,   # K6
    2819478.727,   # K7
    np.nan,        # K8
    589104.255,    # K9
    3903167.450,   # K10
    3273480.602,   # K11
    3109962.229    # Ultimate
]

optimized = [
    462682.615,    # K1
    468832.418,    # K2
    915589.583,    # K3
    837425.598,    # K4
    811193.369,    # K5
    3317442.301,   # K6
    3707963.533,   # K7
    np.nan,        # K8
    3762620.583,   # K9
    3839403.452,   # K10
    3094640.072,   # K11
    4473520.228    # Ultimate
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