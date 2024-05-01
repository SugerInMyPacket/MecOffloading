package compare.gaco;

import unload_decision.Chromosome;
import utils.NumUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class GACO {
    int popSize; // 种群大小
    double crossoverRate; // 交叉概率
    double mutationRate; // 变异概率
    int geneLength; // 染色体的基因长度

    double bound1; // unload 搜索空间边界
    double[] bounds2;
    double[][] bounds3; // freq 搜索空间边界
    double[][] bounds4;

    Chromosome_GACO[] chromosomes;   // 种群（染色体）

    // Pareto前沿  --> 记录的是目双标值
    List<double[]> paretoFront;
    // pareto解对应的pos
    List<double[]> paretoFrontPos;

    Random random;
    public List<double[]> getParetoFront() {
        return paretoFront;
    }

    public List<double[]> getParetoFrontPos() {
        return paretoFrontPos;
    }

    public GACO(int popSize, double crossoverRate, double mutationRate, int geneLength,
                double bound1, double[][] bounds3, double[][] bounds4) {
        random = new Random();

        this.popSize = popSize;
        this.crossoverRate = crossoverRate;
        this.mutationRate = mutationRate;
        this.geneLength = geneLength;

        this.bound1 = bound1;
        this.bounds3 = bounds3;
        this.bounds4 = bounds4;

        this.paretoFront = new ArrayList<>();
        this.paretoFrontPos = new ArrayList<>();

        this.chromosomes = new Chromosome_GACO[popSize];

        for (int i = 0; i < popSize; i++) {
            chromosomes[i] = new Chromosome_GACO(geneLength, bound1, bounds3, bounds4);
            // 更新 pareto 解

            // calculateFitness();
            updateParetoFront(chromosomes[i].gene, chromosomes[i].fitness);
        }

    }

    // 计算种群中每个染色体的适应度
    public void calculateFitness() {
        for (Chromosome_GACO c : chromosomes) {
            // c.fitness = ObjFunctionGA.evaluate(currFreqAllocArr, c.gene);
            c.fitness = ObjFunc_GACO.evaluate(c.geneUnload, c.geneRatio, c.geneFreqLocal, c.geneFreqRemote);
        }
    }

    public Chromosome_GACO selectParent() {
        // 计算种群总适应度
        double totalFitness = Arrays.stream(chromosomes).mapToDouble(c -> c.fitness[0]).sum();
        // 随机选择因子
        double selectFactor = new Random().nextDouble() * totalFitness;

        // 遍历种群，选择符合条件的个体
        double runningFitness = 0.0;
        for (Chromosome_GACO c : chromosomes) {
            runningFitness += c.fitness[0];
            if (runningFitness >= selectFactor) {
                return c;
            }
        }

        // 默认情况返回最后一个个体
        return chromosomes[popSize - 1];
    }

    public Chromosome_GACO crossoverOp(Chromosome_GACO parentA, Chromosome_GACO parentB) {
        Chromosome_GACO child = new Chromosome_GACO(geneLength, bound1, bounds3, bounds4);

        // 单点交叉
        int crossoverPoint = new Random().nextInt(geneLength);

        for (int i = 0; i < geneLength; i++) {
            if(i < crossoverPoint) {
                child.geneUnload[i] = parentA.geneUnload[i];
                child.geneRatio[i] = parentA.geneRatio[i];
                child.geneFreqLocal[i] = parentA.geneFreqLocal[i];
                child.geneFreqRemote[i] = parentA.geneFreqRemote[i];
            }
            else {
                child.geneUnload[i] = parentB.geneUnload[i];
                child.geneRatio[i] = parentB.geneRatio[i];
                child.geneFreqLocal[i] = parentB.geneFreqLocal[i];
                child.geneFreqRemote[i] = parentB.geneFreqRemote[i];
            }
        }

        return child;
    }

    public void mutateOp(Chromosome_GACO c) {
        // 单点变异方式
        for (int i = 0; i < geneLength; i++) {
            if(Math.random() < mutationRate) {
                c.geneUnload[i] = new Random().nextDouble() * bound1;
                c.geneRatio[i] = NumUtil.random(0.0, 1.0);
                c.geneFreqLocal[i] =  100 + random.nextDouble() * 400;
                c.geneFreqRemote[i] = 100 + random.nextDouble() * 400;
            }
        }
    }

    public void optimizeChromosomes(int numGenerations) {
        Chromosome_GACO currBestChromosome = null;
        for (int generation = 0; generation < numGenerations; generation++) {
            // 计算每个染色体的适应度
            calculateFitness();

            // 创建新种群
            Chromosome_GACO[] newChromosomes = new Chromosome_GACO[popSize];

            // 生成下一代种群
            for (int i = 0; i < popSize; i += 2) {
                // 选择
                Chromosome_GACO parentA = selectParent();
                Chromosome_GACO parentB = selectParent();

                // 交叉
                if(Math.random() < crossoverRate) {
                    Chromosome_GACO childA = crossoverOp(parentA, parentB);
                    Chromosome_GACO childB = crossoverOp(parentA, parentB);

                    // TODO：变异
                    mutateOp(childA);
                    mutateOp(childB);

                    newChromosomes[i] = childA;
                    newChromosomes[i + 1] = childB;
                } else {
                    newChromosomes[i] = parentA;
                    newChromosomes[i + 1] = parentB;
                }

                // 检查边界  （是否进考虑加载变异中判断即可）
                for (int j = 0; j < geneLength; j++) {
                    if(newChromosomes[i].geneUnload[j] >= bound1) {
                        newChromosomes[i].geneUnload[j] = bound1 - 1;
                    }
                    if(newChromosomes[i].geneUnload[j] < -1) {
                        newChromosomes[i].geneUnload[j] = -1;
                    }

                    if(newChromosomes[i].geneRatio[j] >= 1.0) {
                        newChromosomes[i].geneRatio[j] = 1.0;
                    }
                    if(newChromosomes[i].geneRatio[j] < 0) {
                        newChromosomes[i].geneRatio[j] = 0;
                    }

                    if(newChromosomes[i].geneFreqLocal[j] > 500) {
                        newChromosomes[i].geneFreqLocal[j] = 500;
                    }
                    if(newChromosomes[i].geneFreqLocal[j] < 100) {
                        newChromosomes[i].geneFreqLocal[j] = 100;
                    }

                    if(newChromosomes[i].geneFreqRemote[j] > 500) {
                        newChromosomes[i].geneFreqRemote[j] = 500;
                    }
                    if(newChromosomes[i].geneFreqRemote[j] < 100) {
                        newChromosomes[i].geneFreqRemote[j] = 100;
                    }
                }

                // fitness 评估
                // double[] newValue = ObjFuncGA2.evaluate(currFreqAllocArr, unloadList);
                double[] newValue = ObjFunc_GACO.evaluate(newChromosomes[i].geneUnload, newChromosomes[i].geneRatio,
                        newChromosomes[i].geneFreqLocal, newChromosomes[i].geneFreqRemote);


                // 更新个体最优解和Pareto前沿
                if (isParetoOptimal(newValue)) {
                    chromosomes[i].fitness = newValue;
                    chromosomes[i].gene = newChromosomes[i].gene;

                    // 记录pareto解对应的gene --> 卸载决策具体数值
                    // addParetoGene(newChromosomes[i].gene);
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
            // 双目标 --->> 双目标 --->> 最大化问题 ( USS, -Energy)
            if (newValue[0] <= solution[0] && newValue[1] <= solution[1]) {
                return false;
            }
        }
        return true;
    }

    // 更新 pareto 解
    public void updateParetoFront(double[] position, double[] value) {
        List<double[]> toRemove = new ArrayList<>();
        List<double[]> toRemovePos = new ArrayList<>();

        int posIndex = 0;
        for (double[] solution : paretoFront) {
            // 如果newValue对于两个目标函数的解都大于已有解，则不是pareto解
            if (solution[0] >= value[0] && solution[1] >= value[1]) {
                // 新解支配已知解，不更新Pareto前沿
                return;
            } else if (solution[0] <= value[0] && solution[1] <= value[1]) {
                // 已知解 支配 新解，需要移除 --> 加入待移除序列
                toRemove.add(solution);
                toRemovePos.add(paretoFrontPos.get(posIndex));
            }

            ++posIndex;
            if (posIndex >= paretoFrontPos.size()) break;
        }
        // 添加新解
        paretoFront.add(value);
        paretoFrontPos.add(position);
        // addParetoPosition(position);
        // 移除非支配解
        paretoFront.removeAll(toRemove);
        paretoFrontPos.removeAll(toRemovePos);
    }

}
