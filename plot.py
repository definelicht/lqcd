import csv, sys
import matplotlib.pyplot as plt
import numpy as np

def filter_data(data):
  accepted = [1, 4, 16, 64, 256, 1024, 2048, 8192]
  keys = list(data.keys())
  for b in keys:
    if int(b) not in accepted:
      del data[b]

def plot_strong(data, includeThermalize=False):

  plt.style.use('ggplot')
  plt.rcParams.update({'font.size': 16})
  fig, ax = plt.subplots()
  fig.set_size_inches(12, 6)
  ax.set_xscale("log", basex=2)
  ax.set_yscale("log")
  ax.set_xlabel("Number of Monte Carlo measurements [1]")
  if not includeThermalize:
    ax.set_ylabel("Time excluding thermalization [s]")
  else:
    ax.set_ylabel("Time including thermalization [s]")
  time = (data["1"]["timeCompute"] if not includeThermalize
          else data["1"]["timeTotal"])
  err = (data["1"]["timeComputeErr"] if not includeThermalize
         else data["1"]["timeTotalErr"])
  ax.errorbar(data["1"]["nRuns"], time, err,
              label="Sequential", marker="o", markersize=10, linewidth=2,
              markeredgewidth=2, linestyle="--")
  del data["1"]
  for i, b in enumerate(sorted(data, key=int)):
    time = (data[b]["timeCompute"] if not includeThermalize
            else data[b]["timeTotal"])
    err = (data[b]["timeComputeErr"] if not includeThermalize
           else data[b]["timeTotalErr"])
    ax.errorbar(data[b]["nRuns"], time, err,
                label="Batch size {}".format(b),
                marker="+" if i >= 6 else "o", markersize=15 if i >= 6 else 10,
                linewidth=2, markeredgewidth=2)
  ax.legend(loc=4, frameon=False, fontsize=15, ncol=2)
  ax.set_xlim([0.9*ax.get_xlim()[0], ax.get_xlim()[1]*1.1])
  ax.set_ylim([0.9*ax.get_ylim()[0], ax.get_ylim()[1]*1.1])

  return fig

def plot_strong_single(data, nRuns):
  nBatches = []
  timeCompute = []
  timeComputeErr = []
  timeTotal = []
  timeTotalErr = []
  for b in data:
    if nRuns not in data[b]["nRuns"]:
      continue
    nBatches.append(int(b))
    timeCompute.append(data[b]["timeCompute"])
    timeComputeErr.append(data[b]["timeComputeErr"])
    timeTotal.append(data[b]["timeTotal"])
    timeTotalErr.append(data[b]["timeTotalErr"])
  nBatches = np.array(nBatches)
  timeCompute = np.array(timeCompute)
  timeComputeErr = np.array(timeComputeErr)
  timeTotal = np.array(timeTotal)
  timeTotalErr = np.array(timeTotalErr)
  fig, ax = plt.subplots()
  print(len(nBatches), len(timeCompute), len(timeComputeErr))
  ax.errorbar(nBatches, timeCompute, timeComputeErr)
  ax.errorbar(nBatches, timeTotal, timeTotalErr)
  return fig

if __name__ == "__main__":

  if len(sys.argv) < 3 or len(sys.argv) > 4:
    print("Usage: <benchmark file> <strong/strong_thermalize/weak> " +
          "[<plot output file>]")
    sys.exit(1)

  benchmarks = []
  with open(sys.argv[1], "r") as inFile:
    readCsv = csv.reader(inFile, delimiter=",")
    readCsv.__next__()
    for row in readCsv:
      benchmarks.append(row)

  data = {}
  for b in sorted(benchmarks, key=lambda t: int(t[0])):
    batchSize = str(b[1])
    if batchSize not in data:
      data[batchSize] = {"nRuns": [], "timeCompute": [], "timeTotal": []}
    data[batchSize]["nRuns"].append(int(b[0]))
    data[batchSize]["timeCompute"].append(float(b[2]))
    data[batchSize]["timeTotal"].append(float(b[3]))

  for b in data:
    nRuns = np.array(data[b]["nRuns"])
    nRunsUnique = np.unique(nRuns)
    timeCompute = np.array(data[b]["timeCompute"])
    timeTotal = np.array(data[b]["timeTotal"])
    timeComputeMean = []
    timeComputeErr = []
    timeTotalMean = []
    timeTotalErr = []
    for n in nRunsUnique:
      indices = nRuns == n
      timeComputeMean.append(np.mean(timeCompute[indices]))
      timeComputeErr.append(np.std(timeCompute[indices]))
      timeTotalMean.append(np.mean(timeTotal[indices]))
      timeTotalErr.append(np.std(timeTotal[indices]))
    data[b]["nRuns"] = nRunsUnique
    data[b]["timeCompute"] = np.array(timeComputeMean)
    data[b]["timeComputeErr"] = np.array(timeComputeErr)
    data[b]["timeTotal"] = np.array(timeTotalMean)
    data[b]["timeTotalErr"] = np.array(timeTotalErr)

  filter_data(data)

  if sys.argv[2] == "strong":
    fig = plot_strong(data, False)
  elif sys.argv[2] == "strong_thermalize":
    fig = plot_strong(data, True)
  elif sys.argv[2] == "strong_single":
    fig = plot_strong_single(data, 2**10)
  elif sys.argv[2] == "weak":
    fig = plot_weak(data)
  else:
    raise ValueError("Invalid plot \"{}\".".format(sys.argv[2]))

  if len(sys.argv) > 3:
    fig.savefig(sys.argv[3], bbox_inches="tight")
  else:
    fig.show()
    input("Press enter to close...")
