import json, sys
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2 or len(sys.argv) > 3:
  print("Usage: <benchmark file> [<plot output file>]")
  sys.exit(1)

with open(sys.argv[1], "r") as inFile:
  benchmarks = json.loads(inFile.read())

nRuns = []
sequential = []
sequentialErr = []
batch = {}
batchErr = {}
for n in sorted(benchmarks["sequential"], key=int):
  nRuns.append(int(n))
  sequential.append(np.mean(benchmarks["sequential"][n]))
  sequentialErr.append(np.std(benchmarks["sequential"][n]))
  for b in benchmarks["batch"]:
    if n not in benchmarks["batch"][b]:
      continue
    if b not in batch:
      batch[b] = []
      batchErr[b] = []
    batch[b].append(np.mean(benchmarks["batch"][b][n]))
    batchErr[b].append(np.std(benchmarks["batch"][b][n]))

nRuns = np.array(nRuns)
sequential = np.array(sequential)
sequentialErr = np.array(sequentialErr)
for b in batch:
  batch[b] = np.array(batch[b])
  batchErr[b] = np.array(batchErr[b])

plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of runs [1]")
ax.set_ylabel("Time [s]")
ax.errorbar(nRuns, sequential, sequentialErr, label="Sequential",
            marker="o", markersize=10, linewidth=2, markeredgewidth=2,
            linestyle="--")
for b in sorted(batch, key=int):
  ax.errorbar(nRuns[len(nRuns)-len(batch[b]):], batch[b], batchErr[b],
              label="Batch size {}".format(b), marker="+", markersize=15,
              linewidth=2, markeredgewidth=2)
ax.legend(loc=2)
ax.set_xlim([ax.get_xlim()[0], ax.get_xlim()[1]*1.1])

if len(sys.argv) > 2:
  fig.savefig(sys.argv[2])
else:
  fig.show()
  input("Press enter to close...")
