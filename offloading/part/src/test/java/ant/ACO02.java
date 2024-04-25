package ant;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ACO02 {

    private static final int NUM_TASKS = 200;
    private static final int NUM_NODES = 50;
    private static final int NUM_ANTS = 20;
    private static final int NUM_ITERATIONS = 100;
    private static final double EVAPORATION_RATE = 0.5;
    private static final double ALPHA = 1.0; // Pheromone importance
    private static final double BETA = 2.0; // Heuristic importance
    private static final double Q = 100.0; // Pheromone deposit constant

    private static final Random random = new Random();

    public static void main(String[] args) {
        ACO02 aco = new ACO02();
        aco.run();
    }

    public void run() {
        List<Ant> ants = new ArrayList<>();
        double[][] pheromones = new double[NUM_TASKS][NUM_NODES];
        double[][] tasks = generateRandomTasks(NUM_TASKS);
        double[][] nodes = generateRandomNodes(NUM_NODES);
        double[][] assignments = new double[NUM_TASKS][2];

        // Initialize pheromone matrix
        initializePheromones(pheromones);

        // Initialize ants
        for (int i = 0; i < NUM_ANTS; i++) {
            ants.add(new Ant(NUM_TASKS));
        }

        // Main loop
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            // Move ants
            for (Ant ant : ants) {
                ant.move(pheromones, tasks, nodes, ALPHA, BETA);
            }

            // Update pheromone levels
            updatePheromones(pheromones, ants);

            // Find best assignment
            double[][] bestAssignment = null;
            double bestCost = Double.MAX_VALUE;
            for (Ant ant : ants) {
                double[][] assignment = ant.getAssignment();
                double cost = calculateCost(assignment);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestAssignment = assignment;
                }
            }

            // Store best assignment
            assignments = bestAssignment;

            // Evaporate pheromones
            evaporatePheromones(pheromones);
        }

        // Output assignments
        System.out.println("Task assignments:");
        for (int i = 0; i < assignments.length; i++) {
            System.out.println("Task " + i + " assigned to Node " + (int) assignments[i][0]);
        }
    }

    private void initializePheromones(double[][] pheromones) {
        for (int i = 0; i < pheromones.length; i++) {
            for (int j = 0; j < pheromones[i].length; j++) {
                pheromones[i][j] = 1.0;
            }
        }
    }

    private double[][] generateRandomTasks(int numTasks) {
        double[][] tasks = new double[numTasks][2];
        for (int i = 0; i < numTasks; i++) {
            tasks[i][0] = random.nextDouble() * 100; // Random delay
            tasks[i][1] = random.nextDouble() * 100; // Random energy consumption
        }
        return tasks;
    }

    private double[][] generateRandomNodes(int numNodes) {
        double[][] nodes = new double[numNodes][2];
        for (int i = 0; i < numNodes; i++) {
            // Random coordinates for demonstration
            nodes[i][0] = random.nextDouble() * 100;
            nodes[i][1] = random.nextDouble() * 100;
        }
        return nodes;
    }

    private void updatePheromones(double[][] pheromones, List<Ant> ants) {
        for (Ant ant : ants) {
            double[][] assignment = ant.getAssignment();
            for (int i = 0; i < assignment.length; i++) {
                int task = i;
                int node = (int) assignment[i][0];
                double pheromoneDelta = Q / calculateCost(assignment);
                pheromones[task][node] += pheromoneDelta;
            }
        }
    }

    private void evaporatePheromones(double[][] pheromones) {
        for (int i = 0; i < pheromones.length; i++) {
            for (int j = 0; j < pheromones[i].length; j++) {
                pheromones[i][j] *= (1 - EVAPORATION_RATE);
            }
        }
    }

    private double calculateCost(double[][] assignment) {
        double totalCost = 0;
        for (double[] task : assignment) {
            totalCost += task[1]; // Energy consumption
        }
        return totalCost;
    }

    static class Ant {
        private double[][] assignment;

        public Ant(int numTasks) {
            this.assignment = new double[numTasks][2];
        }

        public void move_(double[][] pheromones, double[][] tasks, double[][] nodes, double alpha, double beta) {
            // Implement ant movement algorithm
            // For demonstration, let's assume a simple assignment approach
            // You may replace this with a more sophisticated algorithm
            // Update assignment based on pheromones, tasks, and nodes
        }

        public void move(double[][] pheromones, double[][] tasks, double[][] nodes, double alpha, double beta) {
            for (int i = 0; i < assignment.length; i++) {
                double minCost = Double.MAX_VALUE;
                int selectedNode = -1;
                for (int j = 0; j < nodes.length; j++) {
                    // 计算任务 i 分配到节点 j 的总成本
                    double cost = alpha * pheromones[i][j] + beta * nodes[j][2]; // nodes[j][2] 表示节点 j 的成本，这里是个示例，你可以根据实际情况修改
                    if (cost < minCost) {
                        minCost = cost;
                        selectedNode = j;
                    }
                }
                // 将任务 i 分配到成本最低的节点
                assignment[i][0] = selectedNode;
                assignment[i][1] = minCost;
            }
        }


        public double[][] getAssignment() {
            return assignment;
        }
    }

}
