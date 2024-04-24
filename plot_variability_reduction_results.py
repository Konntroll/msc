# %%

import matplotlib.pyplot as plt
import numpy as np

# %%

# Bar chart of overall results for feature subsetting

results = [
    [437, 157, 426, 141, 101, 206, 936],
    [404, 139, 416, 107, 88, 185, 933],
    [377, 132, 401, 93, 87, 181, 926],
    [347, 120, 361, 58, 68, 120, 873]
]
catq = ['SWTQ8', 'SNTQ14', 'TJOQ13', 'SSTQ5', 'SNQ12', 'SNQ15', 'MJOQ12']
techniques = ['With aliases preserved', 'With numbered standardised aliases', 'With a common alias', 'Without aliases']

plt.figure(figsize=(8, 6))

num_sequences = len(results)
bar_width = 0.2
index = np.arange(len(catq))

for i, sequence in enumerate(results):
    plt.bar(index + i * bar_width, sequence, bar_width, label=techniques[i])

plt.xlabel('Category and question')
plt.ylabel('Number of queries in the set')
plt.title('Variability reduction results for feature subsetting')
plt.legend()
plt.xticks(index + (num_sequences / 2) * bar_width, catq)
plt.yticks(np.arange(0, max(max(seq) for seq in results) + 100, 100))
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

# Bar chart of overall results for basic formatting

results = [
  [13454, 4412, 1768, 621, 4043, 913, 2044],
  [8678, 3200, 1212, 459, 1412, 760, 1421],
  [2744, 742, 675, 396, 460, 436, 1241]
]
catq = ['SWTQ8', 'SNTQ14', 'TJOQ13', 'SSTQ5', 'SNQ12', 'SNQ15', 'MJOQ12']
techniques = ['Original queries', 'Set of original queries', 'Set of formatted queries']

plt.figure(figsize=(8, 6))

num_sequences = len(results)
bar_width = 0.2
index = np.arange(len(catq))

for i, sequence in enumerate(results):
    plt.bar(index + i * bar_width, sequence, bar_width, label=techniques[i])

plt.xlabel('Category and question')
plt.ylabel('Number of queries in the set')
plt.title('Variability reduction results for basic formatting')
plt.legend()
plt.xticks(index + (num_sequences / 2) * bar_width, catq)
# plt.yticks(np.arange(0, max(max(seq) for seq in results) + 100, 100))
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

# Bar chart for overall AST representation results

results = [
  [1003, 365, 405, 192, 207, 208, 1043]
]
catq = ['SWTQ8', 'SNTQ14', 'TJOQ13', 'SSTQ5', 'SNQ12', 'SNQ15', 'MJOQ12']
techniques = ['Query ASTs']

plt.figure(figsize=(8, 6))

num_sequences = len(results)
bar_width = 0.35
index = np.arange(len(catq))

for i, sequence in enumerate(results):
    plt.bar(index + (i + num_sequences / 2)  * bar_width, sequence, bar_width, label=techniques[i])

plt.xlabel('Category and question')
plt.ylabel('Number of queries in the set')
plt.title('Variability reduction results for AST representation')
plt.legend()
plt.xticks(index + (num_sequences / 2) * bar_width, catq)
plt.yticks(np.arange(0, max(max(seq) for seq in results) + 100, 100))
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

# Bar chart for overall optimisation results

results = [
  [395, 221, 210, 176, 115, 141, 412]
]
catq = ['SWTQ8', 'SNTQ14', 'TJOQ13', 'SSTQ5', 'SNQ12', 'SNQ15', 'MJOQ12']
techniques = ['Optimised query ASTs']

plt.figure(figsize=(8, 6))

num_sequences = len(results)
bar_width = 0.35
index = np.arange(len(catq))

for i, sequence in enumerate(results):
    plt.bar(index + (i + num_sequences / 2)  * bar_width, sequence, bar_width, label=techniques[i])

plt.xlabel('Category and question')
plt.ylabel('Number of queries in the set')
plt.title('Variability reduction results for optimised query ASTs')
plt.legend()
plt.xticks(index + (num_sequences / 2) * bar_width, catq)
plt.yticks(np.arange(0, max(max(seq) for seq in results) + 100, 100))
plt.grid(True)
plt.tight_layout()
plt.show()