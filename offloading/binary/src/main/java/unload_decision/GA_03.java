package unload_decision;

import resource_allocation.ObjFunction;
import resource_allocation.Particle2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class GA_03 {

    int popSize; // 种群大小
    double crossoverRate; // 交叉概率
    double mutationRate; // 变异概率
    int geneLength; // 染色体的基因长度
    int bound;

    Chromosome2[] chromosomes;   // 种群（染色体）

    // Pareto前沿  --> 记录的是目双标值
    List<double[]> paretoFront;
    // pareto解对应的gene
    List<int[]> paretoFrontGene;

    Random random;

    List<Integer> currFreqAllocArr;  // 资源分配

    public List<double[]> getParetoFront() {
        return paretoFront;
    }


    public List<int[]> getParetoFrontGene() {
        return paretoFrontGene;
    }

    public GA_03(int popSize, double crossoverRate, double mutationRate,
                 int geneLength, int bound,
                 List<Integer> currInputFreqAllocArr) {

        this.popSize = popSize;
        this.crossoverRate = crossoverRate;
        this.mutationRate = mutationRate;
        this.geneLength = geneLength;
        this.bound = bound;
        random = new Random();

        this.chromosomes = new Chromosome2[popSize];

        this.paretoFront = new ArrayList<>();
        this.paretoFrontGene = new ArrayList<>();

        // 初始化资源分配
        currFreqAllocArr = new ArrayList<>();
        for (int i = 0; i < geneLength; i++) {
            currFreqAllocArr.add(currInputFreqAllocArr.get(i));
        }

        // 初始化染色体属性
        for (int i = 0; i < popSize; i++) {
            chromosomes[i] = new Chromosome2(geneLength, bound, currFreqAllocArr);
            // 更新 pareto 解
            updateParetoFront(chromosomes[i].gene, chromosomes[i].fitness);
        }
    }

    /**
     * 初始化种群
     * @param unloadDecisionArr ： 卸载决策数组
     */
    public void initPopulation(List<Integer> unloadDecisionArr) {
        // 根据初始卸载决策，初始化染色体的基因序列
        for (int i = 0; i < popSize; i++) {
            for (int j = 0; j < geneLength; j++) {
                chromosomes[i].gene[j] = unloadDecisionArr.get(j);
            }
        }
    }

    // 计算种群中每个染色体的适应度
    public void calculateFitness() {
        for (Chromosome2 c : chromosomes) {
            c.fitness = ObjFunctionGA.evaluate(currFreqAllocArr, c.gene);
        }
    }

    /**
     * 选择操作 ===> 轮盘赌策略
     * 2、考虑锦标赛算法等
     * Note: 这里仅考虑 USS (fitness[0]) 来选择
     * @return
     */
    public Chromosome2 selectParent() {
        // 计算种群总适应度
        double totalFitness = Arrays.stream(chromosomes).mapToDouble(c -> c.fitness[0]).sum();
        // 随机选择因子
        double selectFactor = new Random().nextDouble() * totalFitness;

        // 遍历种群，选择符合条件的个体
        double runningFitness = 0.0;
        for (Chromosome2 c : chromosomes) {
            runningFitness += c.fitness[0];
            if (runningFitness >= selectFactor) {
                return c;
            }
        }

        // 默认情况返回最后一个个体
        return chromosomes[popSize - 1];
    }

    /**
     * TODO: 交叉操作
     * @param parentA
     * @param parentB
     * @return
     */
    public Chromosome2 crossoverOp(Chromosome2 parentA, Chromosome2 parentB) {
        Chromosome2 child = new Chromosome2(geneLength, bound, currFreqAllocArr);

        // 单点交叉
        int crossoverPoint = new Random().nextInt(geneLength);

        for (int i = 0; i < geneLength; i++) {
            if(i < crossoverPoint) child.gene[i] = parentA.gene[i];
            else child.gene[i] = parentB.gene[i];
        }

        return child;
    }

    // TODO: 变异操作
    public void mutateOp(Chromosome2 chromosome) {
        // 单点变异方式
        for (int i = 0; i < geneLength; i++) {
            if(Math.random() < mutationRate) {
                chromosome.gene[i] = new Random().nextInt(bound);
            }
        }
    }

    /**
     * 染色体迭代优化过程
     */
    public void optimizeChromosomes(int numGenerations) {
        Chromosome2 currBestChromosome = null;
        for (int generation = 0; generation < numGenerations; generation++) {
            // 计算每个染色体的适应度
            calculateFitness();

            // 创建新种群
            Chromosome2[] newChromosomes = new Chromosome2[popSize];

            // 生成下一代种群
            for (int i = 0; i < popSize; i += 2) {
                // 选择
                Chromosome2 parentA = selectParent();
                Chromosome2 parentB = selectParent();

                // 交叉
                if(Math.random() < crossoverRate) {
                    Chromosome2 childA = crossoverOp(parentA, parentB);
                    Chromosome2 childB = crossoverOp(parentA, parentB);

                    // 变异
                    mutateOp(parentA);
                    mutateOp(parentB);

                    newChromosomes[i] = childA;
                    newChromosomes[i + 1] = childB;
                } else {
                    newChromosomes[i] = parentA;
                    newChromosomes[i + 1] = parentB;
                }

                // 检查边界  （是否进考虑加载变异中判断即可）
                for (int j = 0; j < geneLength; j++) {
                    if(newChromosomes[i].gene[j] >= bound) {
                        newChromosomes[i].gene[j] = bound - 1;
                    }
                    if(newChromosomes[i].gene[j] < -1) {
                        newChromosomes[i].gene[j] = -1;
                    }
                }

                // 评估
                double[] newValue = ObjFunctionGA.evaluate(currFreqAllocArr, newChromosomes[i].gene);

                // 更新个体最优解和Pareto前沿
                if (isParetoOptimal(newValue)) {
                    chromosomes[i].fitness = newValue;
                    chromosomes[i].gene = newChromosomes[i].gene;

                    // 记录pareto解对应的gene --> 卸载决策具体数值
                    addParetoGene(newChromosomes[i].gene);
                    // 更新 pareto
                    updateParetoFront(newChromosomes[i].gene, newValue);
                }

            }

            // 染色体数组换代
            chromosomes = newChromosomes;

        }
        // cloneBestChromosome(bestChromosome, currBestChromosome);
    }

    // 判断是否为 pareto 解
    private boolean isParetoOptimal(double[] newValue) {
        // 遍历已知pareto解
        for (double[] solution : paretoFront) {
            // 双目标 --->> 最大化问题 ( USS, -Energy)
            if (newValue[0] <= solution[0] && newValue[1] <= solution[1]) {
                return false;
            }
        }
        return true;
    }

    private void addParetoGene(int[] gene) {
        int[] newParetoPos = new int[geneLength];
        for (int i = 0; i < geneLength; i++) {
            newParetoPos[i] = gene[i];
        }

        paretoFrontGene.add(newParetoPos);
    }

    // 更新 pareto 解
    private void updateParetoFront(int[] gene, double[] value) {
        List<double[]> toRemove = new ArrayList<>();
        List<int[]> toRemovePos = new ArrayList<>();

        int posIndex = 0;
        for (double[] solution : paretoFront) {
            // note：最大化问题
            // 如果newValue对于两个目标函数的解都大于已有解，则不是pareto解
            if (solution[0] >= value[0] && solution[1] >= value[1]) {
                // 新解支配已知解，不更新Pareto前沿
                return;
            } else if (solution[0] <= value[0] && solution[1] <= value[1]) {
                // 已知解 支配 新解，需要移除 --> 加入待移除序列
                toRemove.add(solution);
                toRemovePos.add(paretoFrontGene.get(posIndex));
            }

            ++posIndex;
        }
        // 添加新解
        paretoFront.add(value);
        paretoFrontGene.add(gene);
        // 移除非支配解
        paretoFront.removeAll(toRemove);
        paretoFrontGene.removeAll(toRemovePos);
    }

}
