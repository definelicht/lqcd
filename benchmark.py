import csv, time
import lqcd_batch, lqcd_single

warmup = 1
measurements = 5
nRuns = [2**x for x in range(5, 17)]
batchSizes = [2**x for x in range(1, 14)]
highestRatio = 2**8

results = []
for n in nRuns:
  print("-- {} runs".format(n))
  if n < 2**11:
    print("  Running sequential")
    print("    Warmup runs...")
    for _ in range(warmup):
      lqcd_single.run_montecarlo(n)
    for i in range(measurements):
      _, timeCompute, timeTotal = lqcd_single.run_montecarlo(n)
      print("    Run {} / {}: {:.4f}/{:.4f} seconds".format(
          i+1, measurements, timeCompute, timeTotal))
      results.append((n, "1", timeCompute, timeTotal))
  print("  Running in batches")
  for b in [x for x in batchSizes if n % x == 0 and n / x <= highestRatio]:
    print("    Batch size {}".format(b))
    print("      Warmup runs...")
    for _ in range(warmup):
      lqcd_batch.run_montecarlo(b, int(n/b))
    for i in range(measurements):
      _, timeCompute, timeTotal = lqcd_batch.run_montecarlo(b, int(n/b))
      print("      Run {} / {}: {:.4f}/{:.4f} seconds".format(
          i+1, measurements, timeCompute, timeTotal))
      results.append((n, b, timeCompute, timeTotal))

with open("benchmark.csv", "w") as outFile:
  writeCsv = csv.writer(outFile)
  writeCsv.writerow(("nRuns", "batchSize", "timeCompute", "timeTotal"))
  for r in results:
    writeCsv.writerow(r)
