package ratio_division;

import enums.AlgorithmParam;
import utils.NumUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SA {

    private static final double LEVY_COEFFICIENT = 1.5;
    private Random random;

    int numDimensions;
    double temperature; // 初始温度
    double coolingRate; // 冷却速率
    int stepsPerTemp; // 每个温度的步数
    double minChange; // 最小变化


    List<Integer> currUnloadArr;
    List<Integer> currFreqArrLocal;
    List<Integer> currFreqArrRemote;


    List<double[]> paretoFront; // Pareto 解集合
    List<double[]> paretoFrontRatioArr; // Pareto 解 Ratio 集合
    // List<double[]> paretoFrontUnloadArrLocal;
    // List<double[]> paretoFrontFreqArrLocal;
    // List<double[]> paretoFrontFreqArrRemote;

    public SA(int numDimensions,
                 double temperature,
                 double coolingRate,
                 int stepsPerTemp,
                 double minChange,
                 List<Integer> currInputUnloadArr,
                 List<Integer> currInputFreqArrLocal,
                 List<Integer> currInputFreqArrRemote) {
        random = new Random();

        this.numDimensions = numDimensions;
        this.temperature = temperature;
        this.coolingRate = coolingRate;
        this.stepsPerTemp = stepsPerTemp;
        this.minChange = minChange;

        this.currUnloadArr = new ArrayList<>();
        this.currFreqArrLocal = new ArrayList<>();
        this.currFreqArrRemote = new ArrayList<>();

        for (int i = 0; i < numDimensions; i++) {
            currUnloadArr.add(currInputUnloadArr.get(i));
            currFreqArrLocal.add(currInputFreqArrLocal.get(i));
            currFreqArrRemote.add(currInputFreqArrRemote.get(i));
        }

        paretoFront = new ArrayList<>();
        paretoFrontRatioArr = new ArrayList<>();
        // paretoFrontFreqArrLocal = new ArrayList<>();
        // paretoFrontFreqArrRemote = new ArrayList<>();
    }

    public void optimize() {
        // 进行多次搜索以收集 Pareto 解
        for (int j = 0; j < 1; j++) {
            double[] currentSolution = generateRandomSolution(numDimensions); // 初始解
            double currentTemperature = temperature; // 当前温度

            // 逐渐降温
            while (currentTemperature > minChange) {
                // 当前温度循环
                for (int i = 0; i < stepsPerTemp; i++) {
                    double[] neighbor = generateNeighbor(currentSolution, numDimensions); // 生成邻居解
                    // 计算能量差
                    double[] deltaE = new double[2];

                    double[] newVal
                            = ObjFuncSA.evaluate(currUnloadArr, neighbor, currFreqArrLocal, currFreqArrRemote);
                    double[] currVal
                            = ObjFuncSA.evaluate(currUnloadArr, currentSolution, currFreqArrLocal, currFreqArrRemote);

                    deltaE[0] = newVal[0] - currVal[0];
                    deltaE[1] = newVal[1] - currVal[1];
                    // 如果 neighbor 是pareto解，以一定概率更新 ？
                    // 如果支配所有解，必须更新

                    // 1、是pareto解，替换
                    // 2、不是pareto解, 以一定概率替换
                    if (isParetoOptimal(newVal)
                            || Math.exp(-Math.abs(deltaE[0]) / currentTemperature) > Math.random()) {
                        currentSolution = neighbor; // 接受新解

                        if (isParetoOptimal(newVal)) {
                            // bestSolution = currentSolution.clone(); // 更新最优解
                            updateParetoFront(neighbor, newVal);
                        }
                    }
                }

                currentTemperature *= coolingRate; // 降温
            }
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

    // 生成随机解
    // Todo： 修改，在0.5的基础上进行扰动（混沌映射 ？）
    private double[] generateRandomSolution(int dimensions) {
        double[] solution = new double[dimensions];
        for (int i = 0; i < dimensions; i++) {
            solution[i] = NumUtil.random(0.1, 0.9); // 在 [0,1] 范围内随机生成
        }
        return solution;
    }

    // 生成邻居解
    private double[] generateNeighbor(double[] solution, int dimensions) {
        double[] neighbor = solution.clone();
        // TODO: 多个维度 ？ or  加入 Levy 飞行扰动
        int index = random.nextInt(dimensions); // 随机选择一个维度

        double change = (random.nextDouble() - 0.5) * minChange; // 在一个小范围内改变
        neighbor[index] = Math.max(0, Math.min(1.0, neighbor[index] + change)); // 确保在 [0,1] 范围内
        return neighbor;
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

    // 将解ratio加入pareto
    private void addParetoPosition(double[] ratios) {
        double[] newParetoRatios = new double[numDimensions];
        for (int i = 0; i < numDimensions; i++) {
            newParetoRatios[i] = ratios[i];
        }

        paretoFrontRatioArr.add(newParetoRatios);
    }

    // 更新 pareto 解
    private void updateParetoFront(double[] ratios, double[] value) {
        // List<double[]> toRemove = new ArrayList<>();
        List<Integer> toRemoveIndex = new ArrayList<>();

        int ptrIndex = 0;
        for (double[] solution : paretoFront) {
            // 最大化问题
            // 如果newValue对于两个目标函数的解都大于已有解，则不是pareto解
            if (solution[0] >= value[0] && solution[1] >= value[1]) {
                // 新解支配已知解，不更新Pareto前沿
                return;
            } else if (solution[0] <= value[0] && solution[1] <= value[1]) {
                // 已知解 支配 新解，需要移除 --> 加入待移除序列
                // toRemove.add(solution);
                // toRemovePos.add(paretoFrontPos.get(posIndex));
                // toRemoveUnload.add(paretoUnloadArr.get(posIndex));

                toRemoveIndex.add(ptrIndex);
            }

            ++ptrIndex;
        }
        // 添加新解
        paretoFront.add(value);
        addParetoPosition(ratios);
        // 添加 Pareto ---- unloadArr 和 freqArr

        // 移除非 pareto 解
        for (int i = 0; i < toRemoveIndex.size(); i++) {
            int del = toRemoveIndex.get(i) - i;
            paretoFront.remove(del);
            // paretoFrontPos.remove(i);
            // paretoUnloadArr.remove(i);
            paretoFrontRatioArr.remove(del);
        }
    }


    public List<double[]> getParetoFront() {
        return paretoFront;
    }

    public List<double[]> getParetoFrontRatioArr() {
        return paretoFrontRatioArr;
    }
}
