package unload_ratio;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DE_11 {
    private static int end_iter = 0;
    // private static double curr_maxUSS_iter = 0.0;
    private static double past_maxUSS_iter = 0.0;
    private static final double LEVY_COEFFICIENT = 1.5;

    // 种群规模（M）：一般介于5n到10n之间
    private int populationSize;
    private int maxGenerations;
    // cr一般取[0,1]，通常取0.5。CR越大，收敛速度越快，但易发生早熟现象
    private double crossoverRate;
    // F(mutationFactor)是缩放因子，
    // F越大，越不容易陷入局部极值点；F越小，越有利于收敛到局部极值点。
    private double mutationFactor;
    // 维度
    private int dims;
    // 边界
    private int maxBound;

    Population2[] population;

    private Random random;

    // Pareto前沿  --> 记录的是目双标值
    List<double[]> paretoFront;
    // pareto解对应的gene
    List<double[]> paretoFrontSolution;


    List<Integer> currFreqAllocArrLocal;  // Local 资源分配
    List<Integer> currFreqAllocArrRemote;  // Remote 资源分配

    public List<double[]> getParetoFront() {
        return paretoFront;
    }

    public List<double[]> getParetoFrontSolution() {
        return paretoFrontSolution;
    }

    public DE_11() {
    }

    public DE_11(int populationSize,
                 int maxGenerations,
                 double crossoverRate,
                 double mutationFactor,
                 int dims,
                 int maxBound,
                 List<Integer> currInputUnloadArr,
                 List<Double> currInputRatioArr,
                 List<Integer> currInputFreqAllocArrLocal,
                 List<Integer> currInputFreqAllocArrRemote) {

        this.populationSize = populationSize;
        this.maxGenerations = maxGenerations;
        this.crossoverRate = crossoverRate;
        this.mutationFactor = mutationFactor;
        this.dims = dims;
        this.maxBound = maxBound;
        this.random = new Random();

        // pareto 解
        this.paretoFront = new ArrayList<>();
        this.paretoFrontSolution = new ArrayList<>();

        // 初始化资源分配
        this.currFreqAllocArrLocal = new ArrayList<>();
        this.currFreqAllocArrRemote = new ArrayList<>();
        for (int i = 0; i < dims; i++) {
            currFreqAllocArrLocal.add(currInputFreqAllocArrLocal.get(i));
            currFreqAllocArrRemote.add(currInputFreqAllocArrRemote.get(i));
        }

        // 初始化种群
        population = new Population2[populationSize];
        for (int i = 0; i < populationSize; i++) {
            if (i < 5) {
                population[i] = new Population2(dims, maxBound, currInputUnloadArr,
                        currInputRatioArr, currFreqAllocArrLocal, currFreqAllocArrRemote);
            } else {
                population[i] = new Population2(dims, maxBound,
                        currFreqAllocArrLocal, currFreqAllocArrRemote);
            }

            // 更新 pareto 解
            updateParetoFront(population[i].solutionAll, population[i].fitness);
        }

    }

    // 变异
    public double[] mutate(int targetIndex) {
        int a, b, c;
        do {
            a = random.nextInt(populationSize);
            b = random.nextInt(populationSize);
            c = random.nextInt(populationSize);
        } while (a == targetIndex || b == targetIndex || c == targetIndex ||
                a == b || a == c || b == c);


        double[] levy_vals = levyFlight(dims);

        double[] mutant = new double[dims * 2];
        for (int i = 0; i < dims; i++) {
            // 变异
            mutant[i] = population[a].solutionAll[i]
                    + mutationFactor * (population[b].solutionAll[i] - population[c].solutionAll[i]);

            mutant[i + dims] = population[a].solutionAll[i + dims]
                    + mutationFactor * (population[b].solutionAll[i + dims] - population[c].solutionAll[i + dims]);

            mutant[i] += levy_vals[i];
            mutant[i + dims] += levy_vals[i];

            // 边界处理
            mutant[i] = Math.max(0, Math.min(maxBound, mutant[i]));
            mutant[i + dims] = Math.max(0.0, Math.min(1.0, mutant[i + dims]));
        }

        return mutant;
    }


    // 交叉
    public double[] crossover(double[] target, double[] mutant) {
        double[] trial = new double[dims * 2];
        int jrand = random.nextInt(dims);

        for (int j = 0; j < dims; j++) {
            // 交叉情况
            if (random.nextDouble() <= crossoverRate || j == jrand) {
                trial[j] = mutant[j];
                trial[j + dims] = mutant[j + dims];
            } else {
                trial[j] = target[j];
                trial[j + dims] = target[j + dims];
            }
        }
        return trial;
    }

    // 选择
    private double[] selection(double[] target, double[] trial) {
        double[] targetObj
                = ObjFuncDE2.evaluate(target, currFreqAllocArrLocal, currFreqAllocArrRemote);
        double[] trialObj
                = ObjFuncDE2.evaluate(trial, currFreqAllocArrLocal, currFreqAllocArrRemote);
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
            // 当前代最大的用户满意度
            double curr_maxUSS_iter = 0.0;
            // 粒子
            for (int i = 0; i < populationSize; i++) {
                // 目标解
                double[] target = population[i].solutionAll;
                // 变异  --- 变异个体
                double[] mutant = mutate(i);
                // 交叉  --- 试验个体
                double[] trial = crossover(target, mutant);

                // 选择  --- Pareto dominate : target || trail
                population[i].solutionAll = selection(target, trial);

                // 重新计算适应度
                double[] newValue
                        = ObjFuncDE2.evaluate(population[i].solutionAll,
                        currFreqAllocArrLocal, currFreqAllocArrRemote);

                // 将 solutionAll 分别赋值给 unload 和 ratio
                for (int j = 0; j < dims; j++) {
                    population[i].solutionUnload[j] = (int) population[i].solutionAll[j];
                    population[i].solutionRatio[j] = population[i].solutionAll[j + dims];
                }

                // 更新 适应度
                population[i].fitness = newValue;

                if (isParetoOptimal(newValue)) {
                    // 更新 pareto 解
                    updateParetoFront(population[i].solutionAll, newValue);

                    curr_maxUSS_iter = Math.max(curr_maxUSS_iter, newValue[0]);
                }
            }


            // if (Math.abs(past_maxUSS_iter - curr_maxUSS_iter) < 0.03){
            //     ++end_iter;
            // } else {
            //     end_iter = 0;
            // }
            //
            // if (end_iter == 3) {
            //     break;
            // }
            // past_maxUSS_iter = Math.max(past_maxUSS_iter, curr_maxUSS_iter);

        }
    }


    // Levy 飞行策略
    private static double[] levyFlight(int dim) {
        Random random = new Random();
        double[] step = new double[dim];
        for (int i = 0; i < dim; i++) {
            double u = random.nextGaussian();
            double v = random.nextGaussian();
            double sigma = Math.pow((Math.abs(u) / Math.abs(v)), (1.0 / LEVY_COEFFICIENT));
            double levy = u * sigma;
            step[i] = levy;
        }
        return step;
    }

    private static double levyFlight() {
        Random random = new Random();
        double u = random.nextGaussian();
        double v = random.nextGaussian();
        double step = u / (Math.pow(Math.abs(v), 1 / LEVY_COEFFICIENT));
        return step;
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
            if (solution1[i] > solution2[i]) {
                betterInOne = true;
            }
        }
        if (solution1[0] > solution2[0]) {
            betterInOne = true;
        }
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

    // 更新pareto集
    private void updateParetoFront(double[] sol, double[] fitness) {
        List<double[]> toRemove = new ArrayList<>();
        List<double[]> toRemovePos = new ArrayList<>();

        int posIndex = 0;
        for (double[] oldVal : paretoFront) {
            // 如果newValue对于两个目标函数的解都大于已有解，则不是pareto解
            if (oldVal[0] >= fitness[0] && oldVal[1] >= fitness[1]) {
                // 新解支配已知解，不更新Pareto前沿
                return;
            } else if (oldVal[0] <= fitness[0] && oldVal[1] <= fitness[1]) {
                // 已知解 支配 新解，需要移除 --> 加入待移除序列
                toRemove.add(oldVal);
                toRemovePos.add(paretoFrontSolution.get(posIndex));
            }

            ++posIndex;
        }

        // 添加新解
        paretoFront.add(fitness);
        paretoFrontSolution.add(sol);
        // addParetoSolution(sol);
        // 移除非支配解
        paretoFront.removeAll(toRemove);
        paretoFrontSolution.removeAll(toRemovePos);

    }


}
