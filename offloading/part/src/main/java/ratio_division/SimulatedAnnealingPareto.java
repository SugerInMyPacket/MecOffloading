package ratio_division;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class SimulatedAnnealingPareto {
    private static final double TEMPERATURE = 1000; // 初始温度
    private static final double COOLING_RATE = 0.003; // 冷却速率
    private static final int STEPS_PER_TEMP = 100; // 每个温度的步数
    private static final double MIN_CHANGE = 0.001; // 最小变化

    private static Random random = new Random();

    public static void main(String[] args) {
        int dimensions = 5; // 多维度数组的维度
        List<double[]> paretoFront = new ArrayList<>(); // Pareto 解集合

        for (int j = 0; j < 100; j++) { // 进行多次搜索以收集 Pareto 解
            double[] currentSolution = generateRandomSolution(dimensions); // 初始解
            double[] bestSolution = currentSolution.clone(); // 最优解
            double currentTemperature = TEMPERATURE; // 当前温度

            while (currentTemperature > MIN_CHANGE) {
                for (int i = 0; i < STEPS_PER_TEMP; i++) {
                    double[] neighbor = generateNeighbor(currentSolution, dimensions); // 生成邻居解
                    double deltaE = evaluate(neighbor)[0] - evaluate(currentSolution)[0]; // 计算能量差
                    double deltaE_uss = evaluate(neighbor)[0] - evaluate(currentSolution)[0]; // 计算能量差
                    double deltaE_energy = evaluate(neighbor)[1] - evaluate(currentSolution)[1]; // 计算能量差

                    // 如果 neighbor 是pareto解，1️以一定概率更新 ？
                    // 如果支配所有解，必须更新
                    if (deltaE < 0 || Math.exp(-deltaE / currentTemperature) > Math.random()) {
                        currentSolution = neighbor; // 接受新解
                        if (dominates(currentSolution, bestSolution)) {
                            bestSolution = currentSolution.clone(); // 更新最优解
                        }
                    }
                }
                currentTemperature *= 1 - COOLING_RATE; // 降温
            }
            paretoFront.add(bestSolution); // 将 Pareto 解加入集合
        }

        System.out.println("Pareto Front:");
        for (double[] solution : paretoFront) {
            System.out.println(Arrays.toString(solution));
        }
    }

    // 生成随机解
    private static double[] generateRandomSolution(int dimensions) {
        double[] solution = new double[dimensions];
        for (int i = 0; i < dimensions; i++) {
            solution[i] = random.nextDouble(); // 在 [0,1] 范围内随机生成
        }
        return solution;
    }

    // 生成邻居解
    private static double[] generateNeighbor(double[] solution, int dimensions) {
        double[] neighbor = solution.clone();
        int index = random.nextInt(dimensions); // 随机选择一个维度
        double change = (random.nextDouble() - 0.5) * MIN_CHANGE; // 在一个小范围内改变
        neighbor[index] = Math.max(0, Math.min(1, neighbor[index] + change)); // 确保在 [0,1] 范围内
        return neighbor;
    }

    // 评估解的质量，这里是简单地将两个目标函数的值组成数组返回
    private static double[] evaluate(double[] solution) {
        double[] objectives = new double[2];
        objectives[0] = objective1(solution);
        objectives[1] = objective2(solution);
        return objectives;
    }

    // 目标函数1
    private static double objective1(double[] solution) {
        // 这里是一个示例函数，具体函数需要根据实际情况替换
        return Math.pow(solution[0] - 0.5, 2) + Math.pow(solution[1] - 0.5, 2);
    }

    // 目标函数2
    private static double objective2(double[] solution) {
        // 这里是一个示例函数，具体函数需要根据实际情况替换
        return Math.pow(solution[2] - 0.3, 2) + Math.pow(solution[3] - 0.7, 2);
    }

    // 判断解1是否支配解2
    private static boolean dominates(double[] solution1, double[] solution2) {
        return solution1[0] <= solution2[0] && solution1[1] <= solution2[1] && (solution1[0] < solution2[0] || solution1[1] < solution2[1]);
    }
}
