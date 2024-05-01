package compare.aco;

import java.util.Random;

public class AntColonyAlgorithm {
    private static final double ALPHA = 1; // 信息素重要程度
    private static final double BETA = 2; // 启发因子重要程度
    private static final double RHO = 0.1; // 信息素挥发率
    private static final double Q = 1; // 信息素增加强度

    private static final Random rand = new Random();

    static double randomInRange(double min, double max) {
        return min + (max - min) * rand.nextDouble();
    }

    static void updatePheromone(Ant[] ants) {
        // 更新信息素
        for (Ant ant : ants) {
            for (int j = 0; j < ant.solution.length; j++) {
                ant.pheromone[j] *= (1 - RHO);
            }
            double[] objectives = ant.objectives;
            for (int j = 0; j < ant.solution.length / 2; j++) {
                ant.pheromone[j] += Q / objectives[0];
                ant.pheromone[ant.solution.length / 2 + j] += Q / objectives[1];
            }
        }
    }

    static double[] solve(int n, int iterations, int antsCount, int k) {
        Ant[] ants = new Ant[antsCount];
        double[] globalBest = null;
        double globalBestFitness = Double.POSITIVE_INFINITY;

        for (int i = 0; i < ants.length; i++) {
            ants[i] = new Ant(n);
            for (int j = 0; j < ants[i].solution.length; j++) {
                ants[i].solution[j] = randomInRange(-1, k);
            }
            ants[i].objectives = ObjectiveFunction.evaluate(ants[i].solution);
            if (ants[i].objectives[0] + ants[i].objectives[1] < globalBestFitness) {
                globalBestFitness = ants[i].objectives[0] + ants[i].objectives[1];
                globalBest = ants[i].solution.clone();
            }
        }

        for (int iter = 0; iter < iterations; iter++) {
            for (Ant ant : ants) {
                for (int i = 0; i < n; i++) {
                    // 用轮盘赌法选择下一个节点
                    double total = 0;
                    for (int j = 0; j < n; j++) {
                        total += Math.pow(ant.pheromone[j], ALPHA) * Math.pow(1.0 / (Math.abs(ant.solution[i] - ant.solution[j]) + 1), BETA);
                    }
                    double randValue = rand.nextDouble();
                    double probSum = 0;
                    int selected = -1;
                    for (int j = 0; j < n; j++) {
                        probSum += Math.pow(ant.pheromone[j], ALPHA) * Math.pow(1.0 / (Math.abs(ant.solution[i] - ant.solution[j]) + 1), BETA) / total;
                        if (probSum >= randValue) {
                            selected = j;
                            break;
                        }
                    }
                    if (selected != -1) {
                        ant.solution[i] = ants[selected].solution[i];
                    }
                }
                ant.objectives = ObjectiveFunction.evaluate(ant.solution);
                if (ant.objectives[0] + ant.objectives[1] < globalBestFitness) {
                    globalBestFitness = ant.objectives[0] + ant.objectives[1];
                    globalBest = ant.solution.clone();
                }
            }
            updatePheromone(ants);
        }
        return globalBest;
    }
}
