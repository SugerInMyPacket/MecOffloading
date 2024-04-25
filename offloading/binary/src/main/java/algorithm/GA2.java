package algorithm;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class GA2 {

    static Random random = new Random();
    static int populationSize = 10; // 种群大小
    static double mutationRate = 0.1; // 变异率

    // 遗传算法更新前k位
    public static List<Integer> updateGenetic(List<Integer> list, int k, int m) {
        // 实现遗传算法更新前k位的逻辑
        List<List<Integer>> population = generatePopulation(k, m);

        // 迭代次数，可根据需要修改
        for (int generation = 0; generation < 100; generation++) {
            List<List<Integer>> newPopulation = new ArrayList<>();

            for (int i = 0; i < populationSize; i++) {
                List<Integer> parent1 = selectIndividual(population);
                List<Integer> parent2 = selectIndividual(population);

                List<Integer> offspring = crossover(parent1, parent2);
                offspring = mutate(offspring, m);

                newPopulation.add(offspring);
            }

            population = newPopulation;
        }

        // 选择最优个体
        List<Integer> bestIndividual = population.get(0);
        for (List<Integer> individual : population) {
            if (fitness(individual, k, m) > fitness(bestIndividual, k, m)) {
                bestIndividual = individual;
            }
        }

        for (int i = 0; i < k; i++) {
            list.set(i, bestIndividual.get(i)); // 更新前k位的值
        }

        return list.subList(0, k); // 返回更新后的前k位列表
    }

    // 生成初始种群
    private static List<List<Integer>> generatePopulation(int k, int m) {
        List<List<Integer>> population = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            List<Integer> individual = new ArrayList<>();
            for (int j = 0; j < k; j++) {
                individual.add(random.nextInt(m));
            }
            population.add(individual);
        }
        return population;
    }

    // 选择个体
    private static List<Integer> selectIndividual(List<List<Integer>> population) {
        int index = random.nextInt(populationSize);
        return population.get(index);
    }

    // 交叉
    private static List<Integer> crossover(List<Integer> parent1, List<Integer> parent2) {
        int crossoverPoint = random.nextInt(parent1.size());
        List<Integer> offspring = new ArrayList<>(parent1.subList(0, crossoverPoint));
        offspring.addAll(parent2.subList(crossoverPoint, parent2.size()));
        return offspring;
    }

    // 变异
    private static List<Integer> mutate(List<Integer> individual, int m) {
        for (int i = 0; i < individual.size(); i++) {
            if (random.nextDouble() < mutationRate) {
                individual.set(i, random.nextInt(m));
            }
        }
        return individual;
    }

    // 计算适应度函数
    private static double fitness(List<Integer> individual, int k, int m) {
        // 这里简单地以个体前 k 位的和作为适应度函数，你可以根据实际情况设计更合适的适应度函数
        double sum = 0;
        for (int i = 0; i < k; i++) {
            sum += individual.get(i);
        }
        return sum;
    }

}
