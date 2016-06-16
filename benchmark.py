import csv, multiprocessing, sys, time
import lqcd_batch, lqcd_single

warmup = 1
measurements = 5
nRuns = [16, 32]
batchSizes = [2, 4]
# nRuns = [2**x for x in range(5, 17)]
# batchSizes = [2**x for x in range(1, 14)]
highestRatio = 2**8

def do_run(conf):
  (n, b, reps) = conf
  ret = []
  print("Running {}/{}...".format(n, b))
  if n == 1:
    for _ in range(warmup):
      lqcd_single.run_montecarlo(n)
    for _ in range(reps):
      _, timeCompute, timeTotal = lqcd_single.run_montecarlo(n)
      ret.append((n, b, timeCompute, timeTotal))
  else:
    for _ in range(warmup):
      lqcd_batch.run_montecarlo(n, int(n/b))
    for _ in range(reps):
      _, timeCompute, timeTotal = lqcd_batch.run_montecarlo(n, int(n/b))
      ret.append((n, b, timeCompute, timeTotal))
  print("Finished {}/{}.".format(n, b))
  return ret

pool = multiprocessing.Pool(int(sys.argv[1]))

jobs = []
for n in nRuns:
  if n < 2**11:
    jobs.append((n, 1, measurements))
  for b in [x for x in batchSizes if n % x == 0 and n / x <= highestRatio]:
    jobs.append((n, b, measurements))

res = pool.map(do_run, jobs)
results = []
for r in res:
  results += r

with open("benchmark.csv", "w") as outFile:
  writeCsv = csv.writer(outFile)
  writeCsv.writerow(("nRuns", "batchSize", "timeCompute", "timeTotal"))
  for r in results:
    writeCsv.writerow(r)
