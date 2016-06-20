# \author Johannes de Fine Licht (johannes@musicmedia.dk)
# \date April 2016
#
# Adapted from:
#  G. Peter Lepage: Lattice QCD for Novices
#  Proceedings of HUGS 98, edited by J.L. Goity, World Scientific (2000)
#  arXiv:hep-lat/0506036

import math, sys, time
import numpy as np

class HarmonicOscillator:

  def __init__(self, length, nCor, a, eps, batchSize, dtype=float):
    self.length = length
    self.nCor = nCor
    self.factor0 = 0.5*a
    self.factor1 = 1.0/a
    self.eps = eps
    self.batchSize = batchSize
    self.dtype = dtype
    self.x = np.zeros((self.length, self.batchSize), dtype=self.dtype)
    self.xBuffer = np.empty(self.batchSize, dtype=self.dtype)
    self.valBuffer = np.empty(self.batchSize, dtype=self.dtype)
    self.diffBuffer = np.empty(self.batchSize, dtype=self.dtype)
    self.undoBuffer = np.empty(self.batchSize, dtype=bool)

  def reset(self):
    self.x.fill(0)

  def __evaluate_action(self, i, dst):
    """Compute the action using a harmonic oscillator potential"""
    dst[:] = (self.factor0 * self.x[i, :]**2 +
              self.factor1 * self.x[i, :] * (self.x[i, :] - self.x[i-1, :] -
                                             self.x[(i+1) % self.length, :]))

  def __rand(self, minVal, maxVal):
    return np.random.uniform(minVal, maxVal, size=self.batchSize)

  def __exp(self, arr):
    return np.exp(arr)

  def sweep(self):
    """Single Metropolis over all replications of the lattice"""
    for i in range(self.x.shape[0]):
      self.xBuffer[:] = self.x[i, :]
      self.__evaluate_action(i, self.valBuffer)
      self.x[i, :] += self.__rand(-self.eps, self.eps)
      self.__evaluate_action(i, self.diffBuffer)
      self.diffBuffer -= self.valBuffer
      self.undoBuffer[:] = self.diffBuffer > 0
      self.diffBuffer[self.undoBuffer] = -self.diffBuffer[self.undoBuffer]
      self.undoBuffer &= self.__exp(self.diffBuffer) < self.__rand(0, 1)
      self.x[i, self.undoBuffer] = self.xBuffer[self.undoBuffer]

  def run(self):
    """Perform enough sweeps to produce a new atcb of decorrelated samples"""
    for _ in range(self.nCor):
      self.sweep()

  def thermalize(self, factor=5):
    """Thermalize the lattices by performing a large number of sweeps"""
    for _ in range(factor):
      self.run()

def accumulate_g(g, x):
  """Compute contributions to the expectation value of G"""
  size = x.shape[0]
  for dist in range(size):
    for i in range(size):
      g[dist] += np.sum(x[i, :] * x[(i+dist) % size, :])

def run_montecarlo(batchSize, nBatches, length=20, nCor=20, a=0.5, eps=1.4,
                   dtype=float):
  """Performs a full evaluation by initializing, thermalizing, running and
     analyzing the system"""
  osc = HarmonicOscillator(length, nCor, a, eps, batchSize, dtype)
  gAvg = np.zeros(osc.length, dtype=osc.dtype)
  osc.reset()
  start = time.time()
  osc.thermalize()
  startCompute = time.time()
  for i in range(nBatches):
    osc.run()
    accumulate_g(gAvg, osc.x)
  stop = time.time()
  timeCompute = stop - startCompute
  timeTotal = stop - start
  gAvg *= 1. / (osc.length * nBatches * osc.batchSize)
  return gAvg, timeCompute, timeTotal

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: <number of runs> <batch size>")
    sys.exit(1)
  nRuns = int(sys.argv[1])
  batchSize = int(sys.argv[2])
  if nRuns % batchSize != 0:
    print("Number of runs must be divisible by batch size.")
    sys.exit(1)
  nBatches = int(nRuns / batchSize)
  gAvg, timeCompute, timeTotal = run_montecarlo(batchSize, nBatches)
  print("Finished in {:.4f} seconds ({:.4f} including thermalization).".format(
       timeCompute, timeTotal))
  print("Resulting expectation values:\n", gAvg)
