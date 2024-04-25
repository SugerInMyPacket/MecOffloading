package demo;

import java.util.Arrays;
import java.util.Random;

public class GeneticAlgorithm {

    private static final int POPULATION_SIZE = 50;
    private static final double MUTATION_RATE = 0.1;
    private static final int NUM_ITERATIONS = 100;
    private static final Random random = new Random();

    public static void main(String[] args) {
        double[][] optimalSolution = findOptimalSolution();
        System.out.println("Optimal Solution: " + Arrays.deepToString(optimalSolution));
    }

    private static double[][] findOptimalSolution() {
        double[][] population = initializePopulation();
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            population = evolvePopulation(population);
        }
        return population;
    }

    private static double[][] initializePopulation() {
        double[][] population = new double[POPULATION_SIZE][2]; // Assuming 2 variables (integer and decimal)
        for (int i = 0; i < POPULATION_SIZE; i++) {
            population[i][0] = random.nextInt(100); // Initialize integer variable randomly
            population[i][1] = random.nextDouble() * 100; // Initialize decimal variable randomly
        }
        return population;
    }

    private static double[][] evolvePopulation(double[][] population) {
        double[][] newPopulation = new double[POPULATION_SIZE][2];
        for (int i = 0; i < POPULATION_SIZE; i++) {
            double[] parent1 = selectParent(population);
            double[] parent2 = selectParent(population);
            double[] child = crossover(parent1, parent2);
            mutate(child);
            newPopulation[i] = child;
        }
        return newPopulation;
    }

    private static double[] selectParent(double[][] population) {
        // Tournament selection
        int tournamentSize = 5;
        double[] bestParent = null;
        double bestFitness = Double.MAX_VALUE;
        for (int i = 0; i < tournamentSize; i++) {
            double[] candidate = population[random.nextInt(POPULATION_SIZE)];
            double fitness = evaluateFitness(candidate);
            if (fitness < bestFitness) {
                bestFitness = fitness;
                bestParent = candidate;
            }
        }
        return bestParent;
    }

    private static double[] crossover(double[] parent1, double[] parent2) {
        double[] child = new double[parent1.length];
        int crossoverPoint = random.nextInt(parent1.length);
        for (int i = 0; i < crossoverPoint; i++) {
            child[i] = parent1[i];
        }
        for (int i = crossoverPoint; i < parent2.length; i++) {
            child[i] = parent2[i];
        }
        return child;
    }

    private static void mutate(double[] child) {
        for (int i = 0; i < child.length; i++) {
            if (random.nextDouble() < MUTATION_RATE) {
                if (i == 0) {
                    child[i] = random.nextInt(100); // Mutate integer variable
                } else {
                    child[i] = random.nextDouble() * 100; // Mutate decimal variable
                }
            }
        }
    }

    private static double evaluateFitness(double[] solution) {
        // Replace with your fitness function
        double fitness = 0;
        // Example fitness function: minimize the sum of squares
        for (double var : solution) {
            fitness += var * var;
        }
        return fitness;
    }
}
