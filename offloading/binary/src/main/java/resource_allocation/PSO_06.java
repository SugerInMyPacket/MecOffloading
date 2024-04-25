package resource_allocation;

import enums.Constants;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class PSO_06 {

    int numParticles; // 粒子数量
    int numDimensions; // 维度数量
    double[][] bounds; // 搜索空间边界
    double inertiaWeight; // 惯性权重 w
    double cognitiveWeight; // 认知权重 σ1
    double socialWeight; // 社会权重  σ2

    // Pareto前沿  --> 记录的是目双标值
    List<double[]> paretoFront;
    // pareto解对应的pos
    List<double[]> paretoFrontPos;

    Particle3[] particles; // 粒子群
    Random random;

    // 当前接收卸载决策
    List<Integer> currUnloadArr;

    List<int[]> paretoUnloadArr;
    List<Integer> currParticleUnloadArr;

    public List<int[]> getParetoUnloadArr() {
        return paretoUnloadArr;
    }

    public List<double[]> getParetoFront() {
        return paretoFront;
    }

    public List<double[]> getParetoFrontPos() {
        return paretoFrontPos;
    }

    public PSO_06(int numParticles, int numDimensions, double[][] bounds,
                  double inertiaWeight, double cognitiveWeight, double socialWeight,
                  List<Integer> currInputUnloadArr) {

        this.numParticles = numParticles;
        this.numDimensions = numDimensions;
        this.bounds = bounds;
        this.inertiaWeight = inertiaWeight;
        this.cognitiveWeight = cognitiveWeight;
        this.socialWeight = socialWeight;

        this.paretoFront = new ArrayList<>();
        this.paretoFrontPos = new ArrayList<>();

        this.paretoUnloadArr = new ArrayList<>();

        this.particles = new Particle3[numParticles];

        this.random = new Random();

        // 初始化卸载决策
        this.currUnloadArr = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            currUnloadArr.add(currInputUnloadArr.get(i));
        }

        this.currParticleUnloadArr = new ArrayList<>(currInputUnloadArr);

        // 初始化粒子属性
        for (int i = 0; i < numParticles; i++) {
            if (i < numParticles / 5) {

            } else {

            }
            particles[i] = new Particle3(numDimensions, bounds, currUnloadArr);
            // 更新 pareto 解
            updateParetoFront(particles[i].pBestPosition, particles[i].pBestFitness);
        }
    }

    public PSO_06(int numParticles, int numDimensions, double[][] bounds,
                  double inertiaWeight, double cognitiveWeight, double socialWeight,
                  List<Integer> currInputUnloadArr, List<Integer> currInputFreqArr) {

        this.numParticles = numParticles;
        this.numDimensions = numDimensions;
        this.bounds = bounds;
        this.inertiaWeight = inertiaWeight;
        this.cognitiveWeight = cognitiveWeight;
        this.socialWeight = socialWeight;

        this.paretoFront = new ArrayList<>();
        this.paretoFrontPos = new ArrayList<>();

        this.paretoUnloadArr = new ArrayList<>();

        this.particles = new Particle3[numParticles];

        this.random = new Random();

        // 初始化卸载决策
        this.currUnloadArr = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            currUnloadArr.add(currInputUnloadArr.get(i));
        }

        this.currParticleUnloadArr = new ArrayList<>(currInputUnloadArr);

        // 初始化粒子属性
        for (int i = 0; i < numParticles; i++) {
            if (i < 5) {
                particles[i] = new Particle3(numDimensions, bounds, currUnloadArr, currInputFreqArr);
            } else {
                particles[i] = new Particle3(numDimensions, bounds, currUnloadArr);
            }
            // particles[i] = new Particle3(numDimensions, bounds, currUnloadArr);
            // 更新 pareto 解
            updateParetoFront(particles[i].pBestPosition, particles[i].pBestFitness);
        }
    }


    public void initParticles(List<Integer> currInputUnloadArr, List<int[]> freqAllocArr) {

    }


    // 优化迭代
    // Note：根据迭代次数前后期，改变权重的变化
    public void optimize(int maxIterations) {
        // System.out.println(bounds[0][0] + "," + bounds[0][1]);
        // 代数
        for (int iter = 0; iter < maxIterations; iter++) {
            // 粒子
            for (int i = 0; i < numParticles; i++) {
                double[] newVelocity = new double[numDimensions];
                double[] newPosition = new double[numDimensions];

                // 更新惯性参数 w （inertiaWeight） [0.4,2]
                inertiaWeight = (1.5 - 0.5) / 2.0
                        + Math.tanh((-4 + 8 * (maxIterations - iter)) / maxIterations) * (1.5 - 0.5) / 2.0;

                // 更新个体认知 cognitiveWeight 和 社会认知 socialWeight  推荐取值范围：[0,4]
                cognitiveWeight = 3.2 - iter * (3.2 - 0.8) / maxIterations;
                socialWeight = 1.2 + iter * (3.4 - 1.2) / maxIterations;


                int sel_index = random.nextInt(paretoFront.size());
                double c1 = random.nextDouble();
                double c2 = random.nextDouble();

                // 维度
                for (int j = 0; j < numDimensions; j++) {
                    // 计算认知和社会分量
                    double cognitiveComponent
                            = cognitiveWeight * c1 * (particles[i].pBestPosition[j] - particles[i].position[j]);
                    // double socialComponent = socialWeight * random.nextDouble() * (paretoFront.get(random.nextInt(paretoFront.size()))[j] - particles[i].position[j]);
                    // double socialComponent = socialWeight * random.nextDouble() * (paretoFrontPos.get(random.nextInt(paretoFront.size()))[j] - particles[i].position[j]);
                    double socialComponent
                            = socialWeight * c2 * (paretoFrontPos.get(sel_index)[j] - particles[i].position[j]);

                    // 更新速度和位置
                    newVelocity[j]
                            = inertiaWeight * particles[i].velocity[j] + cognitiveComponent + socialComponent;
                    if (newVelocity[j] > 500) {
                        newVelocity[j] = 500;
                    }

                    newPosition[j] = particles[i].position[j] + newVelocity[j];

                    // 确保新位置在边界内
                    if (newPosition[j] < bounds[j][0]) {
                        newPosition[j] = bounds[j][0];
                    } else if (newPosition[j] > bounds[j][1]) {
                        newPosition[j] = bounds[j][1];
                    }
                }

                // 适应度值评估
                // currParticleUnloadArr = new ArrayList<>(currUnloadArr);
                double[] newValue
                        = ObjFuncPSO2.evaluate(particles[i].currParticleUnloadList, newPosition);
                // 在评估过程中，修改了 unloadArr，若当前解为pareto解，
                // 应该记录修改后的unloadArr，作为下一大轮迭代的决策数据
                currParticleUnloadArr = new ArrayList<>(particles[i].currParticleUnloadList);
                int[] temp = new int[numDimensions];
                for (int j = 0; j < numDimensions; j++) {
                    temp[j] = currParticleUnloadArr.get(j);
                }
                // addParetoUnloadArr4CurrP(temp);

                // 更新个体最优解和Pareto前沿
                if (isParetoOptimal(newValue)) {
                    particles[i].pBestFitness = newValue;
                    particles[i].pBestPosition = newPosition;

                    // addParetoUnloadArr4CurrP(temp);
                    // addParetoPosition(newPosition);
                    // 更新 pareto
                    updateParetoFront(newPosition, newValue);
                }

                double alpha = Constants.FITNESS_RATIO;
                double sumOfTwoObj_New = alpha * newValue[0] - alpha * newValue[1];
                double sumOfTwoObj_Old
                        = alpha * particles[i].pBestFitness[0] - alpha * particles[i].pBestFitness[1];
                // 更新粒子速度和位置
                // TODO：改为 pareto ？ or 加权双目标->替换
                // if(sumOfTwoObj_New > sumOfTwoObj_Old) {
                //     particles[i].velocity = newVelocity;
                //     particles[i].position = newPosition;
                //     particles[i].pBestFitness = newValue;
                // }

                // particles[i].velocity = newVelocity;
                // particles[i].position = newPosition;
            }
        }

    }


    // 判断是否为 pareto 解
    private boolean isParetoOptimal(double[] newValue) {
        // 遍历已知pareto解
        for (double[] solution : paretoFront) {
            // keynote：双目标 --->> 最大化问题 ( USS, -Energy)
            if (newValue[0] <= solution[0] && newValue[1] <= solution[1]) {
                return false;
            }
        }
        return true;
    }

    // 将解position加入pareto
    private void addParetoPosition(double[] pos) {
        double[] newParetoPos = new double[numDimensions];
        for (int i = 0; i < numDimensions; i++) {
            newParetoPos[i] = pos[i];
        }

        paretoFrontPos.add(newParetoPos);
    }

    public void addParetoUnloadArr4CurrP(int[] unloadArr) {
        paretoUnloadArr.add(unloadArr);
    }

    // 更新 pareto 解
    private void updateParetoFront(double[] position, double[] value) {
        List<double[]> toRemove = new ArrayList<>();
        List<double[]> toRemovePos = new ArrayList<>();
        List<int[]> toRemoveUnload = new ArrayList<>();

        List<Integer> toRemoveIndex = new ArrayList<>();

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
                toRemovePos.add(paretoFrontPos.get(posIndex));
                toRemoveUnload.add(paretoUnloadArr.get(posIndex));

                toRemoveIndex.add(posIndex);
            }

            ++posIndex;
        }
        // 添加新解
        paretoFront.add(value);
        // paretoFrontPos.add(position);
        addParetoPosition(position);
        int[] tempUnloadArr = new int[currUnloadArr.size()];
        for (int i = 0; i < currUnloadArr.size(); i++) {
            tempUnloadArr[i] = currParticleUnloadArr.get(i);
        }
        // paretoUnloadArr.add(tempUnloadArr);
        addParetoUnloadArr4CurrP(tempUnloadArr);

        // 移除非支配解
        paretoFront.removeAll(toRemove);
        paretoFrontPos.removeAll(toRemovePos);
        paretoUnloadArr.removeAll(toRemoveUnload);
        // for (int i = 0; i < toRemoveIndex.size(); i++) {
        //     int del = toRemoveIndex.get(i);
        //     paretoFrontPos.remove(del);
        //     paretoUnloadArr.remove(del);
        // }
    }

}
