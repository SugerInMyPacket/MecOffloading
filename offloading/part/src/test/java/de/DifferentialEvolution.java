package de;

import java.util.Random;

public class DifferentialEvolution {
    // 种群规模（M）：一般介于5n到10n之间
    private int populationSize;
    private int maxGenerations;
    // cr一般取[0,1]，通常取0.5。CR越大，收敛速度越快，但易发生早熟现象
    private double crossoverRate;
    // F(mutationFactor)是缩放因子，
    // F越大，越不容易陷入局部极值点；F越小，越有利于收敛到局部极值点。
    private double mutationFactor;
    // 维度
    private int dimensions;
    // 边界
    private int maxBound;

    private Random random;

    public DifferentialEvolution(int populationSize,
                                 int maxGenerations,
                                 double crossoverRate,
                                 double mutationFactor,
                                 int dimensions,
                                 int maxBound) {
        this.populationSize = populationSize;
        this.maxGenerations = maxGenerations;
        this.crossoverRate = crossoverRate;
        this.mutationFactor = mutationFactor;
        this.dimensions = dimensions;
        this.maxBound = maxBound;
        this.random = new Random();
    }

    // Define objective functions
    private double[] evaluate(double[] solution) {
        // Define your objective functions here
        double[] objectives = new double[2];
        objectives[0] = objective1(solution);
        objectives[1] = objective2(solution);
        return objectives;
    }

    // Objective function 1
    private double objective1(double[] solution) {
        // Define your first objective function here
        // Example: return some function of solution
        return 0.0;
    }

    // Objective function 2
    private double objective2(double[] solution) {
        // Define your second objective function here
        // Example: return some function of solution
        return 0.0;
    }

    // Initialize population
    private double[][] initializePopulation() {
        double[][] population = new double[populationSize][dimensions];
        for (int i = 0; i < populationSize; i++) {
            for (int j = 0; j < dimensions; j++) {
                // Integer values from 0 to maxBound
                population[i][j] = random.nextInt(maxBound + 1);
            }
        }
        return population;
    }

    // Differential mutation
    private double[] mutate(double[][] population, int targetIndex) {
        int a, b, c;
        // do {
        //     a = random.nextInt(populationSize);
        // } while (a == targetIndex);
        // do {
        //     b = random.nextInt(populationSize);
        // } while (b == targetIndex || b == a);
        // do {
        //     c = random.nextInt(populationSize);
        // } while (c == targetIndex || c == a || c == b);

        // a = b = c = targetIndex;
        do {
            a = random.nextInt(populationSize);
            b = random.nextInt(populationSize);
            c = random.nextInt(populationSize);
        } while (a == targetIndex || b == targetIndex || c == targetIndex ||
                a == b || a == c || b == c);


        double[] mutant = new double[dimensions];
        for (int i = 0; i < dimensions; i++) {
            mutant[i] = population[a][i] + mutationFactor * (population[b][i] - population[c][i]);
            // Ensure within bounds
            mutant[i] = Math.max(0, Math.min(maxBound, mutant[i]));
        }
        return mutant;
    }

    // Crossover
    private double[] crossover(double[] target, double[] mutant) {
        double[] trial = new double[dimensions];
        int jrand = random.nextInt(dimensions);
        for (int j = 0; j < dimensions; j++) {
            if (random.nextDouble() <= crossoverRate || j == jrand) {
                trial[j] = mutant[j];
            } else {
                trial[j] = target[j];
            }
        }
        return trial;
    }

    // Select the best among target and trial
    private double[] selection(double[] target, double[] trial) {
        double[] targetObj = evaluate(target);
        double[] trialObj = evaluate(trial);
        // Pareto dominance check
        if (isParetoDominant(trialObj, targetObj)) {
            return trial;
        } else {
            return target;
        }
    }

    // Check if a solution dominates another
    private boolean isParetoDominant(double[] solution1, double[] solution2) {
        // Pareto dominance check
        // solution1 dominates solution2 if it is equal or better in all objectives
        // and better in at least one
        boolean betterInOne = false;
        for (int i = 0; i < solution1.length; i++) {
            if (solution1[i] < solution2[i]) {
                return false;
            } else if (solution1[i] > solution2[i]) {
                betterInOne = true;
            }
        }
        return betterInOne;
    }

    // Main optimization loop
    public double[][] optimize() {
        double[][] population = initializePopulation();
        for (int generation = 0; generation < maxGenerations; generation++) {
            for (int i = 0; i < populationSize; i++) {
                double[] target = population[i];
                double[] mutant = mutate(population, i);
                double[] trial = crossover(target, mutant);
                population[i] = selection(target, trial);
            }
        }
        return population;
    }

    public static void main(String[] args) {
        int populationSize = 100;
        int maxGenerations = 100;
        double crossoverRate = 0.8;
        double mutationFactor = 0.5;
        int dimensions = 20; // Number of dimensions for the problem
        int maxBound = 10; // Maximum value for each dimension

        DifferentialEvolution de
                = new DifferentialEvolution(populationSize, maxGenerations, crossoverRate,
                mutationFactor, dimensions, maxBound);
        double[][] paretoFront = de.optimize();

        // Print Pareto front
        for (double[] solution : paretoFront) {
            System.out.print("Solution: ");
            for (double value : solution) {
                System.out.print(value + " ");
            }
            System.out.println();
        }
    }
}
