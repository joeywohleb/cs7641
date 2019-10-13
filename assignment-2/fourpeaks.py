import sys
import os
import time

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import shared.ConvergenceTrainer as ConvergenceTrainer

from array import array


"""
Commandline parameter(s):
   none
"""

N = 200
T = N/5
fill = [2] * N
ranges = array('i', fill)

ef = FourPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)

hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

times = ""
print "RHC:"
for x in range(20):
    start = time.time()
    iterations = (x + 1) * 2500
    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, iterations)
    fit.train()
    print(str(ef.value(rhc.getOptimal())))
    end = time.time()
    times += "\n%0.03f" % (end - start)
print(times)

times = ""
print "SA:"
for x in range(20):
    start = time.time()
    iterations = (x + 1) * 2500
    sa = SimulatedAnnealing(1E11, .95, hcp)
    fit = FixedIterationTrainer(sa, iterations)
    fit.train()
    print(str(ef.value(sa.getOptimal())))
    end = time.time()
    times += "\n%0.03f" % (end - start)
print(times)

times = ""
print "GA:"
for x in range(20):
    start = time.time()
    iterations = (x + 1) * 2500
    ga = StandardGeneticAlgorithm(200, 100, 10, gap)
    fit = FixedIterationTrainer(ga, iterations)
    fit.train()
    print(str(ef.value(ga.getOptimal())))
    end = time.time()
    times += "\n%0.03f" % (end - start)
print(times)

times = ""
print "MIMIC:"
for x in range(20):
    start = time.time()
    iterations = (x + 1) * 2500
    mimic = MIMIC(200, 20, pop)
    fit = FixedIterationTrainer(mimic, iterations)
    fit.train()
    print(str(ef.value(mimic.getOptimal())))
    end = time.time()
    times += "\n%0.03f" % (end - start)
print(times)
