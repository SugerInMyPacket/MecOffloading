package unload_decision;

import utils.FormatData;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DE {

    // 种群规模（M）：一般介于5n到10n之间
    private int populationSize;
    private int maxGenerations;
    // cr一般取[0,1]，通常取0.5。CR越大，收敛速度越快，但易发生早熟现象
    private double crossoverRate;
    // F(mutationFactor)是缩放因子，
    // F越大，越不容易陷入局部极值点；F越小，越有利于收敛到局部极值点。
    private double mutationFactor;
    // 维度
    private int dimensions;
    // 边界
    private int maxBound;

    Population[] population;

    private Random random;

    // Pareto前沿  --> 记录的是目双标值
    List<double[]> paretoFront;
    // pareto解对应的gene
    List<int[]> paretoFrontSolution;

    List<Double> currUnloadRatioArr;  // 卸载比例
    List<Integer> currFreqAllocArrLocal;  // Local 资源分配
    List<Integer> currFreqAllocArrRemote;  // Remote 资源分配


    public List<double[]> getParetoFront() {
        return paretoFront;
    }

    public List<int[]> getParetoFrontSolution() {
        return paretoFrontSolution;
    }

    public DE() {
    }

    public DE(int populationSize,
              int maxGenerations,
              double crossoverRate,
              double mutationFactor,
              int dimensions,
              int maxBound,
              List<Double> currInputUnloadRatioArr,
              List<Integer> currInputFreqAllocArrLocal,
              List<Integer> currInputFreqAllocArrRemote) {

        this.populationSize = populationSize;
        this.maxGenerations = maxGenerations;
        this.crossoverRate = crossoverRate;
        this.mutationFactor = mutationFactor;
        this.dimensions = dimensions;
        this.maxBound = maxBound;
        this.random = new Random();


        // pareto 解
        this.paretoFront = new ArrayList<>();
        this.paretoFrontSolution = new ArrayList<>();


        // 初始化卸载比例
        this.currUnloadRatioArr = new ArrayList<Double>();
        for (int i = 0; i < dimensions; i++) {
            currUnloadRatioArr.add(currInputUnloadRatioArr.get(i));
        }

        // 初始化资源分配
        this.currFreqAllocArrLocal = new ArrayList<>();
        this.currFreqAllocArrRemote = new ArrayList<>();
        for (int i = 0; i < dimensions; i++) {
            currFreqAllocArrLocal.add(currInputFreqAllocArrLocal.get(i));
            currFreqAllocArrRemote.add(currInputFreqAllocArrRemote.get(i));
        }

        // 初始化种群
        population = new Population[populationSize];
        for (int i = 0; i < populationSize; i++) {
            population[i]
                    = new Population(dimensions, maxBound, currUnloadRatioArr,
                    currFreqAllocArrLocal, currFreqAllocArrRemote);

            // 更新 pareto 解
            updateParetoFront(population[i].solution, population[i].fitness);
        }

    }


    public DE(int populationSize,
              int maxGenerations,
              double crossoverRate,
              double mutationFactor,
              int dimensions,
              int maxBound,
              List<Integer> currInputUnloadArr,
              List<Double> currInputUnloadRatioArr,
              List<Integer> currInputFreqAllocArrLocal,
              List<Integer> currInputFreqAllocArrRemote) {

        this.populationSize = populationSize;
        this.maxGenerations = maxGenerations;
        this.crossoverRate = crossoverRate;
        this.mutationFactor = mutationFactor;
        this.dimensions = dimensions;
        this.maxBound = maxBound;
        this.random = new Random();


        // pareto 解
        this.paretoFront = new ArrayList<>();
        this.paretoFrontSolution = new ArrayList<>();


        // 初始化卸载比例
        this.currUnloadRatioArr = new ArrayList<Double>();
        for (int i = 0; i < dimensions; i++) {
            currUnloadRatioArr.add(currInputUnloadRatioArr.get(i));
        }

        // 初始化资源分配
        this.currFreqAllocArrLocal = new ArrayList<>();
        this.currFreqAllocArrRemote = new ArrayList<>();
        for (int i = 0; i < dimensions; i++) {
            currFreqAllocArrLocal.add(currInputFreqAllocArrLocal.get(i));
            currFreqAllocArrRemote.add(currInputFreqAllocArrRemote.get(i));
        }

        // 初始化种群
        population = new Population[populationSize];
        for (int i = 0; i < populationSize; i++) {

            if (i < 5) {
                population[i] = new Population(dimensions, maxBound,
                        currInputUnloadArr, currUnloadRatioArr,
                        currFreqAllocArrLocal, currFreqAllocArrRemote);
            } else {
                population[i] = new Population(dimensions, maxBound, currUnloadRatioArr,
                        currFreqAllocArrLocal, currFreqAllocArrRemote);
            }

            // 更新 pareto 解
            updateParetoFront(population[i].solution, population[i].fitness);
        }

    }


    // 计算种群中每个解的适应度
    public void calculateFitness() {
        for (Population p : population) {
            p.fitness = ObjFuncGA.evaluate(p.solution, currUnloadRatioArr,
                    currFreqAllocArrLocal, currFreqAllocArrRemote);
        }
    }

    // 变异
    private int[] mutate(int targetIndex) {
        int a, b, c;
        do {
            a = random.nextInt(populationSize);
            b = random.nextInt(populationSize);
            c = random.nextInt(populationSize);
        } while (a == targetIndex || b == targetIndex || c == targetIndex ||
                a == b || a == c || b == c);


        double[] mutant = new double[dimensions];
        for (int i = 0; i < dimensions; i++) {
            // 变异
            mutant[i] = population[a].solution[i]
                    + mutationFactor * (population[b].solution[i] - population[c].solution[i]);

            // 边界处理
            mutant[i] = Math.max(0, Math.min(maxBound, mutant[i]));
        }


        int[] res = FormatData.getIntArr(mutant);
        return res;
        // return FormatData.getIntArr(mutant);
    }


    // 交叉
    private int[] crossover(int[] target, int[] mutant) {
        int[] trial = new int[dimensions];
        int jrand = random.nextInt(dimensions);

        for (int j = 0; j < dimensions; j++) {
            // 交叉情况
            if (random.nextDouble() <= crossoverRate || j == jrand) {
                trial[j] = mutant[j];
            } else {
                trial[j] = target[j];
            }
        }
        return trial;
    }

    // Select the best among target and trial
    private int[] selection(int[] target, int[] trial) {
        double[] targetObj
                = ObjFuncDE.evaluate(target, currUnloadRatioArr, currFreqAllocArrLocal, currFreqAllocArrRemote);
        double[] trialObj
                = ObjFuncDE.evaluate(trial, currUnloadRatioArr, currFreqAllocArrLocal, currFreqAllocArrRemote);
        // Pareto dominance check
        // if (isParetoOptimal(trialObj))
        if (isParetoDominant(targetObj, trialObj)) {
            // 如果试验解 trail 支配 target解
            return trial;
        } else {
            return target;
        }
    }


    // Optimization loop
    public void optimize() {
        // 迭代
        for (int generation = 0; generation < maxGenerations; generation++) {
            for (int i = 0; i < populationSize; i++) {
                // 目标解
                int[] target = population[i].solution;
                // 变异  --- 变异个体
                int[] mutant = mutate(i);
                // 交叉  --- 试验个体
                int[] trial = crossover(target, mutant);

                // 选择  --- Pareto dominate : target || trail
                population[i].solution = selection(target, trial);

                // 重新计算适应度
                double[] newValue
                        = ObjFuncDE.evaluate(population[i].solution, currUnloadRatioArr,
                        currFreqAllocArrLocal, currFreqAllocArrRemote);

                population[i].fitness = newValue;

                if (isParetoOptimal(newValue)) {
                    // 更新 pareto 解
                    updateParetoFront(population[i].solution, newValue);
                }
            }

            // 更新所有粒子的适应度
            // calculateFitness();
        }
    }

    // 判断 solution1 与 solution2 的支配关系
    private boolean isParetoDominant(double[] solution1, double[] solution2) {
        // Pareto支配关系
        // solution1 dominates solution2 if it is equal or better in all objectives
        // and better in at least one
        boolean betterInOne = false;
        for (int i = 0; i < solution1.length; i++) {
            // if (solution1[i] < solution2[i]) {
            //     return false;
            // } else if (solution1[i] > solution2[i]) {
            //     betterInOne = true;
            // }
            if (solution1[i] < solution2[i]) {
                betterInOne = true;
            }
        }
        if (solution1[0] < solution2[0]) betterInOne = true;
        return betterInOne;
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


    private void addParetoSolution(int[] s) {
        int[] newParetoSol = new int[dimensions];
        for (int i = 0; i < dimensions; i++) {
            newParetoSol[i] = s[i];
        }

        paretoFrontSolution.add(newParetoSol);
    }


    // 更新 pareto 解
    private void updateParetoFront(int[] sol, double[] newVal) {
        List<double[]> toRemove = new ArrayList<>();
        List<int[]> toRemovePos = new ArrayList<>();

        int posIndex = 0;
        for (double[] oldVal : paretoFront) {
            // 如果newValue对于两个目标函数的解都大于已有解，则不是pareto解
            if (oldVal[0] >= newVal[0] && oldVal[1] >= newVal[1]) {
                // 新解支配已知解，不更新Pareto前沿
                return;
            } else if (oldVal[0] <= newVal[0] && oldVal[1] <= newVal[1]) {
                // 已知解 支配 新解，需要移除 --> 加入待移除序列
                toRemove.add(oldVal);
                toRemovePos.add(paretoFrontSolution.get(posIndex));
            }

            ++posIndex;
        }
        // 添加新解
        paretoFront.add(newVal);
        // paretoFrontSolution.add(sol);
        addParetoSolution(sol);
        // 移除非支配解
        paretoFront.removeAll(toRemove);
        paretoFrontSolution.removeAll(toRemovePos);
    }
}
