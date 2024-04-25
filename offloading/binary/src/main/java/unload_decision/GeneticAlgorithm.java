package unload_decision;

import java.util.Arrays;
import java.util.Random;

public class GeneticAlgorithm {

    // 定义个体类
    static class Individual {
        int[] chromosome; // 染色体
        double fitness; // 适应度

        // 构造函数
        public Individual(int chromosomeLength) {
            this.chromosome = new int[chromosomeLength];
            this.fitness = 0.0;

            // 初始化染色体，可以根据实际问题进行调整
            for (int i = 0; i < chromosomeLength; i++) {
                // TODO: 假设是二进制编码，可以根据实际情况进行修改
                this.chromosome[i] = new Random().nextInt(9);
            }
        }
    }

    // 遗传算法的主要类
    static class GeneticAlgorithmOptimizer {
        int populationSize; // 种群大小
        double crossoverRate; // 交叉概率
        double mutationRate; // 变异概率
        int chromosomeLength; // 染色体长度
        Individual[] population; // 种群

        // 构造函数
        public GeneticAlgorithmOptimizer(int populationSize, double crossoverRate, double mutationRate, int chromosomeLength) {
            this.populationSize = populationSize;
            this.crossoverRate = crossoverRate;
            this.mutationRate = mutationRate;
            this.chromosomeLength = chromosomeLength;
            this.population = new Individual[populationSize];

            // 初始化种群
            for (int i = 0; i < populationSize; i++) {
                population[i] = new Individual(chromosomeLength);
            }
        }

        // 评估适应度的方法，根据实际问题进行实现
        private double evaluateFitness(int[] chromosome) {
            // 这里可以是需要优化的目标函数
            // 例如：计算染色体中1的数量
            int countOnes = 0;
            for (int gene : chromosome) {
                if (gene == 1) {
                    countOnes++;
                }
            }
            return countOnes;
        }

        // 计算种群中每个个体的适应度
        private void calculateFitness() {
            for (Individual individual : population) {
                individual.fitness = evaluateFitness(individual.chromosome);
            }
        }

        // 选择操作，采用轮盘赌选择法
        private Individual selectParent() {
            // 计算总适应度
            double totalFitness = Arrays.stream(population).mapToDouble(individual -> individual.fitness).sum();
            double rand = new Random().nextDouble() * totalFitness;

            // 遍历种群，选择符合条件的个体
            double runningFitness = 0.0;
            for (Individual individual : population) {
                runningFitness += individual.fitness;
                if (runningFitness >= rand) {
                    return individual;
                }
            }

            // 如果未能选择到个体，则返回最后一个个体
            return population[population.length - 1];
        }

        // 交叉操作，采用单点交叉
        private Individual crossover(Individual parent1, Individual parent2) {
            Individual child = new Individual(chromosomeLength);
            int crossoverPoint = new Random().nextInt(chromosomeLength);

            for (int i = 0; i < chromosomeLength; i++) {
                if (i < crossoverPoint) {
                    child.chromosome[i] = parent1.chromosome[i];
                } else {
                    child.chromosome[i] = parent2.chromosome[i];
                }
            }

            return child;
        }

        // 变异操作，采用单点变异
        private void mutate(Individual individual) {
            for (int i = 0; i < chromosomeLength; i++) {
                if (Math.random() < mutationRate) {
                    // 变异操作，简单地将染色体中的一个基因取反
                    individual.chromosome[i] = (individual.chromosome[i] == 0) ? 1 : 0;
                }
            }
        }

        // 遗传算法的主要迭代过程
        public void optimize(int numGenerations) {
            for (int generation = 0; generation < numGenerations; generation++) {
                // 计算适应度
                calculateFitness();

                // 创建新种群
                Individual[] newPopulation = new Individual[populationSize];

                // 生成下一代种群
                for (int i = 0; i < populationSize; i += 2) {
                    // 选择父母
                    Individual parent1 = selectParent();
                    Individual parent2 = selectParent();

                    // 交叉
                    if (Math.random() < crossoverRate) {
                        Individual child1 = crossover(parent1, parent2);
                        Individual child2 = crossover(parent1, parent2);

                        // 变异
                        mutate(child1);
                        mutate(child2);

                        newPopulation[i] = child1;
                        newPopulation[i + 1] = child2;
                    } else {
                        newPopulation[i] = parent1;
                        newPopulation[i + 1] = parent2;
                    }
                }

                // 更新种群
                population = newPopulation;

                // 输出每代的最优解
                Individual bestIndividual = Arrays.stream(population).max((i1, i2) -> Double.compare(i1.fitness, i2.fitness)).orElse(null);
                System.out.println("Generation " + (generation + 1) + ": Best Fitness = " + bestIndividual.fitness +
                        ", Chromosome = " + Arrays.toString(bestIndividual.chromosome));
            }
        }
    }

    // 主函数
    public static void main(String[] args) {
        int populationSize = 20; // 种群大小
        double crossoverRate = 0.8; // 交叉概率
        double mutationRate = 0.01; // 变异概率
        int chromosomeLength = 10; // 染色体长度
        int numGenerations = 50; // 迭代代数

        // 创建遗传算法优化器
        GeneticAlgorithmOptimizer optimizer = new GeneticAlgorithmOptimizer(populationSize, crossoverRate, mutationRate, chromosomeLength);

        // 执行遗传算法
        optimizer.optimize(numGenerations);
    }
}

