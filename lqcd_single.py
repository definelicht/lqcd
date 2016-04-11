# \author Johannes de Fine Licht (johannes@musicmedia.dk)
# \date April 2016
#
# Adapted from:
#  G. Peter Lepage: Lattice QCD for Novices
#  Proceedings of HUGS 98, edited by J.L. Goity, World Scientific (2000)
#  arXiv:hep-lat/0506036

import numpy as np
import math, sys

class HarmonicOscillator:

  def __init__(self, length, nCor, a, eps, dtype=float):
    self.length = length
    self.nCor = nCor
    self.factor0 = 0.5*a
    self.factor1 = 1.0/a
    self.eps = eps
    self.dtype = dtype
    self.x = np.zeros(self.length, dtype=self.dtype)

  def reset(self):
    self.x[:] = 0

  def compute_value(self, i):
    return (self.factor0*self.x[i]**2 +
            self.factor1*self.x[i]*(self.x[i] - self.x[i-1] -
                                    self.x[(i+1) % self.length]))

  def sweep(self):
    for i in range(self.x.size):
      posOld = self.x[i]
      valOld = self.compute_value(i)
      self.x[i] = self.x[i] + np.random.uniform(-self.eps, self.eps)
      diff = self.compute_value(i) - valOld
      # TODO: builtin exp is faster than numpy on scalars. Switch to numpy when
      # vectorizing.
      if diff > 0 and math.exp(-diff) < np.random.uniform():
        self.x[i] = posOld

  def run(self):
    for _ in range(self.nCor):
      self.sweep()

  def thermalize(self, factor=5):
    for _ in range(factor):
      self.run()

def accumulate_g(g, x):
  size = x.size
  for dist in range(size):
    for i in range(size):
      g[dist] += x[i] * x[(i+dist) % size]

def run_montecarlo(nRuns, length=20, nCor=20, a=0.5, eps=1.4, dtype=float):
  osc = HarmonicOscillator(length, nCor, a, eps, dtype)
  gAvg = np.zeros(osc.length, dtype=osc.dtype)
  osc.reset()
  osc.thermalize()
  for i in range(nRuns):
    osc.run()
    accumulate_g(gAvg, osc.x)
  gAvg *= 1. / (osc.length * nRuns)
  return gAvg

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: <number of runs>")
    sys.exit(1)
  nRuns = int(sys.argv[1])
  gAvg = run_montecarlo(nRuns)
  print('Average G:\n', gAvg)
