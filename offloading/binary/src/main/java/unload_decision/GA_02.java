package unload_decision;

import config.InitFrame;
import config.RevisePolicy;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import enums.Constants;
import lombok.extern.slf4j.Slf4j;
import utils.FormatData;
import utils.Formula;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

@Slf4j(topic = "GA_02")
public class GA_02 {

    int popSize; // 种群大小
    double crossoverRate; // 交叉概率
    double mutationRate; // 变异概率
    int geneLength; // 染色体的基因长度

    Chromosome[] chromosomes;   // 种群（染色体）

    Chromosome bestChromosome;  // 迭代 numGenerations 次后得到的最优解

    List<Integer> currFreqAllocArr;  // 资源分配


    public GA_02() { }

    public GA_02(int popSize, double crossoverRate, double mutationRate, int geneLength) {
        this.popSize = popSize;
        this.crossoverRate = crossoverRate;
        this.mutationRate = mutationRate;
        this.geneLength = geneLength;

        this.chromosomes = new Chromosome[popSize];
        for (int i = 0; i < popSize; i++) {
            chromosomes[i] = new Chromosome(geneLength);
        }

        this.bestChromosome = new Chromosome(geneLength);

    }

    public GA_02(int popSize, int geneLength, List<Integer> inputFreqAllocArr) {
        this.popSize = popSize;
        this.geneLength = geneLength;
        int size = inputFreqAllocArr.size();
        this.currFreqAllocArr = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            currFreqAllocArr.add(inputFreqAllocArr.get(i));
        }
    }

    public GA_02(int popSize, double crossoverRate, double mutationRate, int geneLength, List<Integer> inputFreqAllocArr) {
        this.popSize = popSize;
        this.crossoverRate = crossoverRate;
        this.mutationRate = mutationRate;
        this.geneLength = geneLength;
        int size = inputFreqAllocArr.size();
        this.currFreqAllocArr = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            currFreqAllocArr.add(inputFreqAllocArr.get(i));
        }

        this.chromosomes = new Chromosome[popSize];
        for (int i = 0; i < popSize; i++) {
            chromosomes[i] = new Chromosome(geneLength);
        }

        this.bestChromosome = new Chromosome(geneLength);
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

    /**
     * TODO: 染色体适应度计算
     * 这里应需要引入 资源分配的变量，来根据优化目标计算适应度
     * 初步可以考虑使用 加权多目标求和
     * @param chromosome : 染色体
     * @return
     */
    public double evaluateFitness(Chromosome chromosome) {
        // return FormatData.getEffectiveValue4Digit(new Random().nextDouble(), 5);

        List<Integer> currChromosomeUnloadArr = new ArrayList<>();
        for (int i = 0; i < geneLength; i++) {
            currChromosomeUnloadArr.add((int) chromosome.gene[i]);
        }

        List<Integer> currChromosomeResAllocArr = new ArrayList<>();
        for (int i = 0; i < geneLength; i++) {
            currChromosomeResAllocArr.add(currFreqAllocArr.get(i));
        }

        List<Task> taskList = InitFrame.getTaskList();
        List<Vehicle> vehicleList = InitFrame.getVehicleList();
        RoadsideUnit rsu = InitFrame.getRSU();
        // 计算uss和energy
        List<Double> currUssList = Formula.getUSS4TaskList(taskList, currChromosomeUnloadArr, currChromosomeResAllocArr);
        List<Double> currEnergyList = Formula.getEnergy4TaskList(taskList, currChromosomeUnloadArr, currChromosomeResAllocArr);


        RevisePolicy.reviseUnloadArr(taskList, vehicleList, rsu, currChromosomeUnloadArr, currChromosomeResAllocArr,
                currUssList, currEnergyList);

        // 修正后重新计算uss和energy
        currUssList = Formula.getUSS4TaskList(taskList, currChromosomeUnloadArr, currChromosomeResAllocArr);
        currEnergyList = Formula.getEnergy4TaskList(taskList, currChromosomeUnloadArr, currChromosomeResAllocArr);

        // 暂定适应度为 ussTotal - theta / energyTotal
        double ussTotal = 0.0;
        double energyTotal = 0.0;
        for (int i = 0; i < geneLength; i++) {
            ussTotal += currUssList.get(i);
            energyTotal += currEnergyList.get(i);
        }

        double fitness = ussTotal / geneLength - Constants.USS_THRESHOLD / (energyTotal / geneLength);

        return FormatData.getEffectiveValue4Digit(1 / fitness, 5);
    }

    // 计算种群中每个染色体的适应度
    public void calculateFitness() {
        for (Chromosome c : chromosomes) {
            c.fitness = evaluateFitness(c);
        }
    }

    /**
     * 选择操作 ===> 轮盘赌策略
     * 2、考虑锦标赛算法等
     * @return
     */
    public Chromosome selectParent() {
        // 计算种群总适应度
        double totalFitness = Arrays.stream(chromosomes).mapToDouble(c -> c.fitness).sum();
        // 随机选择因子
        double selectFactor = new Random().nextDouble() * totalFitness;

        // 遍历种群，选择符合条件的个体
        double runningFitness = 0.0;
        for (Chromosome c : chromosomes) {
            runningFitness += c.fitness;
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
    public Chromosome crossoverOp(Chromosome parentA, Chromosome parentB) {
        Chromosome child = new Chromosome(geneLength);

        // 单点交叉
        int crossoverPoint = new Random().nextInt(geneLength);

        for (int i = 0; i < geneLength; i++) {
            if(i < crossoverPoint) child.gene[i] = parentA.gene[i];
            else child.gene[i] = parentB.gene[i];
        }

        return child;
    }

    // TODO: 变异操作
    public void mutateOp(Chromosome chromosome) {
        // 单点变异方式
        for (int i = 0; i < geneLength; i++) {
            if(Math.random() < mutationRate) {
                chromosome.gene[i] = new Random().nextInt(10);
            }
        }
    }

    /**
     * 染色体迭代优化过程
     */
    public void optimizeChromosomes(int numGenerations) {
        Chromosome currBestChromosome = null;
        for (int generation = 0; generation < numGenerations; generation++) {
            // 计算每个染色体的适应度
            calculateFitness();

            // 创建新种群
            Chromosome[] newChromosomes = new Chromosome[popSize];

            // 生成下一代种群
            for (int i = 0; i < popSize; i += 2) {
                // 选择
                Chromosome parentA = selectParent();
                Chromosome parentB = selectParent();

                // 交叉
                if(Math.random() < crossoverRate) {
                    Chromosome childA = crossoverOp(parentA, parentB);
                    Chromosome childB = crossoverOp(parentA, parentB);

                    // 变异
                    mutateOp(parentA);
                    mutateOp(parentB);

                    newChromosomes[i] = childA;
                    newChromosomes[i + 1] = childB;
                } else {
                    newChromosomes[i] = parentA;
                    newChromosomes[i + 1] = parentB;
                }

            }

            chromosomes = newChromosomes;

            // 找出每代的最优个体
            currBestChromosome =
                    Arrays.stream(chromosomes)
                            .max((c1, c2) -> Double.compare(c1.fitness, c2.fitness))
                            .orElse(null);
            // 打印
            // log.info("Generation " + (generation + 1) + ": Best Fitness = " + currBestChromosome.fitness
            //         + ", Chromosome = " + Arrays.toString(currBestChromosome.gene));

            // cloneBestChromosome(bestChromosome, currBestChromosome);
            // TODO: 设置边界，修改越界元素
        }
        cloneBestChromosome(bestChromosome, currBestChromosome);
    }


    public void cloneBestChromosome(Chromosome result, Chromosome best) {
        // 深拷贝
        // int len = best.gene.length;
        for (int i = 0; i < geneLength; i++) {
            result.gene[i] = best.gene[i];
        }
        result.fitness = best.fitness;

    }

    // 得到当前最优的个体
    public Chromosome getCurrBestChromosome() {
        return bestChromosome;
    }

}
