import json, time
import lqcd_batch, lqcd_single

warmup = 3
measurements = 3
nRuns = [100, 200, 500, 1000, 2000, 5000, 10000]
batchSizes = list(nRuns)

results = {"batch": {}, "sequential": {}}
for b in batchSizes:
  results["batch"][str(b)] = {}
for n in nRuns:
  print("-- {} runs".format(n))
  print("  Running sequential")
  print("    Warmup runs...")
  for _ in range(warmup):
    lqcd_single.run_montecarlo(n)
  results["sequential"][str(n)] = []
  for i in range(measurements):
    tBegin = time.time()
    lqcd_single.run_montecarlo(n)
    tElapsed = time.time() - tBegin
    print("    Run {} / {}: {} seconds".format(i+1, measurements, tElapsed))
    results["sequential"][str(n)].append(tElapsed)
    # outFile.write("sequential,{},,{}\n".format(n, tElapsed))
  print("  Running in batches")
  results["batch"][str(n)] = {}
  for b in [x for x in batchSizes if n % x == 0]:
    print("    Batch size {}".format(b))
    print("      Warmup runs...")
    for _ in range(warmup):
      lqcd_batch.run_montecarlo(b, int(n/b))
    results["batch"][str(b)][str(n)] = []
    for i in range(measurements):
      tBegin = time.time()
      lqcd_batch.run_montecarlo(b, int(n/b))
      tElapsed = time.time() - tBegin
      print("      Run {} / {}: {} seconds".format(i+1, measurements,
                                                   tElapsed))
      results["batch"][str(b)][str(n)].append(tElapsed)
      # outFile.write("batch,{},{},{}\n".format(n, b, tElapsed))

serialized = json.dumps(results)
with open("benchmark.json", "w") as outFile:
  outFile.write(serialized)
