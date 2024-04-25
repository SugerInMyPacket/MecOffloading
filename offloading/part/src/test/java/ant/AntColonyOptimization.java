package ant;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class AntColonyOptimization {
    private static final int NUM_ANTS = 20;
    private static final int NUM_ITERATIONS = 100;
    static final int NUM_OBJECTIVES = 2;
    private static final int NUM_DIMENSIONS = 10;
    private static final double EVAPORATION_RATE = 0.5;
    private static final double ALPHA = 1.0; // Pheromone importance
    private static final double BETA = 2.0; // Heuristic importance
    private static final double Q = 100.0; // Pheromone deposit constant

    private static final Random random = new Random();

    public void run() {
        List<Ant> ants = new ArrayList<>();
        double[][] pheromones = new double[NUM_DIMENSIONS][NUM_DIMENSIONS];
        double[][] distances = new double[NUM_DIMENSIONS][NUM_DIMENSIONS];
        double[][] objectives = new double[NUM_DIMENSIONS][NUM_OBJECTIVES];
        double[][] paretoFront = new double[NUM_ITERATIONS][NUM_OBJECTIVES];

        // Initialize pheromone matrix and distance matrix
        initializePheromones(pheromones);
        initializeDistances(distances);

        // Initialize ants
        for (int i = 0; i < NUM_ANTS; i++) {
            ants.add(new Ant(NUM_DIMENSIONS));
        }

        // Main loop
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            // Move ants
            for (Ant ant : ants) {
                ant.move(pheromones, distances, ALPHA, BETA);
            }

            // Update pheromone levels
            updatePheromones(pheromones, ants);

            // Find and store Pareto front
            for (int i = 0; i < NUM_DIMENSIONS; i++) {
                objectives[i] = ants.get(i).getObjectives();
            }
            paretoFront[iter] = findParetoFront(objectives);

            // Evaporate pheromones
            evaporatePheromones(pheromones);
        }

        // Output Pareto front
        System.out.println("Pareto Front:");
        for (double[] point : paretoFront) {
            for (double value : point) {
                System.out.print(value + " ");
            }
            System.out.println();
        }
    }

    private void initializePheromones(double[][] pheromones) {
        for (int i = 0; i < NUM_DIMENSIONS; i++) {
            for (int j = 0; j < NUM_DIMENSIONS; j++) {
                pheromones[i][j] = 1.0;
            }
        }
    }

    private void initializeDistances(double[][] distances) {
        for (int i = 0; i < NUM_DIMENSIONS; i++) {
            for (int j = 0; j < NUM_DIMENSIONS; j++) {
                distances[i][j] = random.nextDouble() * 10; // Random distances for demonstration
            }
        }
    }

    private void updatePheromones(double[][] pheromones, List<Ant> ants) {
        for (Ant ant : ants) {
            double[][] tour = ant.getTour();
            double[] objectives = ant.getObjectives();
            for (int i = 0; i < tour.length - 1; i++) {
                int from = (int) tour[i][0];
                int to = (int) tour[i + 1][0];
                double pheromoneDelta = Q / objectives[0];
                pheromones[from][to] += pheromoneDelta;
            }
        }
    }

    private void evaporatePheromones(double[][] pheromones) {
        for (int i = 0; i < NUM_DIMENSIONS; i++) {
            for (int j = 0; j < NUM_DIMENSIONS; j++) {
                pheromones[i][j] *= (1 - EVAPORATION_RATE);
            }
        }
    }

    private double[] findParetoFront(double[][] objectives) {
        List<double[]> paretoFront = new ArrayList<>();
        for (double[] point : objectives) {
            boolean isPareto = true;
            for (double[] otherPoint : objectives) {
                if (point != otherPoint && isDominating(otherPoint, point)) {
                    isPareto = false;
                    break;
                }
            }
            if (isPareto) {
                paretoFront.add(point);
            }
        }
        return paretoFront.get(random.nextInt(paretoFront.size()));
    }

    private boolean isDominating(double[] pointA, double[] pointB) {
        for (int i = 0; i < pointA.length; i++) {
            if (pointA[i] < pointB[i]) {
                return false;
            }
        }
        return true;
    }
}
