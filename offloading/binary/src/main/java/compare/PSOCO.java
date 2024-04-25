package compare;

import enums.Constants;
import resource_allocation.ObjFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class PSOCO {

    int numParticles; // 粒子数量
    int numDimensions; // 维度数量  2*d
    int bound1; // unload 搜索空间边界
    double[][] bounds2; // freq 搜索空间边界
    double inertiaWeight; // 惯性权重 w
    double cognitiveWeight; // 认知权重 σ1
    double socialWeight; // 社会权重  σ2

    Particle_PSOCO[] particles;

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

    public Particle_PSOCO[] getParticles() {
        return particles;
    }

    public PSOCO(int numParticles, int numDimensions, int bound1, double[][] bounds2,
                 double inertiaWeight, double cognitiveWeight, double socialWeight) {

        this.numParticles = numParticles;
        this.numDimensions = numDimensions;
        this.bound1 = bound1;
        this.bounds2 = bounds2;
        this.inertiaWeight = inertiaWeight;
        this.cognitiveWeight = cognitiveWeight;
        this.socialWeight = socialWeight;
        this.paretoFront = new ArrayList<>();
        this.paretoFrontPos = new ArrayList<>();
        this.particles = new Particle_PSOCO[numParticles];
        this.random = new Random();

        for (int i = 0; i < numParticles; i++) {
            particles[i] = new Particle_PSOCO(numDimensions, bound1, bounds2);
            // 更新 pareto 解
            updateParetoFront(particles[i].pBestPosition, particles[i].pBestFitness);
        }
    }

    // 优化迭代
    public void optimize(int maxIterations) {
        // 代数
        for (int iter = 0; iter < maxIterations; iter++) {
            // 粒子
            for (int i = 0; i < numParticles; i++) {
                double[] newVelocity = new double[numDimensions * 2];
                double[] newPosition = new double[numDimensions * 2];

                int[] unloadArr = new int[numDimensions];
                double[] freqArr = new double[numDimensions];

                // 更新惯性参数 w （inertiaWeight） [0.4,2]
                inertiaWeight = 2.0 - iter * (2.0 - 0.4) / maxIterations;

                // 更新个体认知 cognitiveWeight 和 社会认知 socialWeight  推荐取值范围：[0,4]
                // cognitiveWeight = 3.2 - iter * (3.2 - 0.8) / maxIterations;
                // socialWeight = 1.2 + iter * (3.4 - 1.2) / maxIterations;

                // Random random = new Random();
                int sel_index = random.nextInt(paretoFront.size());
                double c1 = random.nextDouble();
                double c2 = random.nextDouble();
                // 维度
                for (int j = 0; j < numDimensions; j++) {
                    int k = j + numDimensions;
                    // 计算认知和社会分量
                    // double cognitiveComponent = cognitiveWeight * random.nextDouble() * (particles[i].pBestPosition[j] - particles[i].position[j]);
                    double cognitiveComponent
                            = cognitiveWeight * c1 * (particles[i].pBestPosition[j] - particles[i].position[j]);
                    // double socialComponent = socialWeight * c1 * (paretoFrontPos.get(sel_index)[j] - particles[i].position[j]);
                    double socialComponent
                            = socialWeight * c2 * (paretoFrontPos.get(sel_index)[j] - particles[i].position[j]);

                    double cognitiveComponent2 = cognitiveWeight * c1 * (particles[i].pBestPosition[k] - particles[i].position[k]);
                    double socialComponent2 = socialWeight * c2 * (paretoFrontPos.get(sel_index)[k] - particles[i].position[k]);

                    // 更新速度和位置
                    newVelocity[j]
                            = inertiaWeight * particles[i].velocity[j] + cognitiveComponent + socialComponent;
                    newPosition[j] = particles[i].position[j] + newVelocity[j];

                    newVelocity[k] = inertiaWeight * particles[i].velocity[k] + cognitiveComponent2 + socialComponent2;
                    newPosition[k] = particles[i].position[k] + newVelocity[k];

                    // 卸载决策的边界检查 --- 确保新位置在边界内
                    if (newPosition[j] < -1) {
                        newPosition[j] = -1;
                    } else if (newPosition[j] > bound1) {
                        newPosition[j] = bound1;
                    }
                    // 资源分配的边界检查
                    if (newPosition[k] < bounds2[j][0]) {
                        newPosition[k] = bounds2[j][0];
                    } else if (newPosition[k] > bounds2[j][1]) {
                        newPosition[k] = bounds2[j][1];
                    }

                    // 赋值
                    unloadArr[j] = (int) newPosition[j];
                    freqArr[j] = newPosition[k];
                }

                // 适应度值评估
                double[] newValue = ObjFunc_PSOCO.evaluate(unloadArr, freqArr);
                // 在评估过程中，修改了 unloadArr，若当前解为pareto解，应该记录修改后的unloadArr，作为下一大轮迭代的决策数据
                for (int j = 0; j < numDimensions; j++) {
                    newPosition[j] = unloadArr[j];
                    newPosition[j + numDimensions] = freqArr[j];
                }

                // 更新个体最优解和Pareto前沿
                if (isParetoOptimal(newValue)) {
                    particles[i].pBestFitness = newValue;
                    particles[i].pBestPosition = newPosition;

                    // addParetoPosition(newPosition);
                    // 更新 pareto
                    updateParetoFront(newPosition, newValue);
                }

                double alpha = Constants.FITNESS_RATIO;
                double sumOfTwoObj_New = alpha * newValue[0] - alpha * newValue[1];
                double sumOfTwoObj_Old = alpha * particles[i].pBestFitness[0] - alpha * particles[i].pBestFitness[1];
                // 更新粒子速度和位置
                // TODO：改为 pareto ？ or 加权双目标->替换
                // if(sumOfTwoObj_New > sumOfTwoObj_Old) {
                particles[i].velocity = newVelocity;
                particles[i].position = newPosition;
                // particles[i].pBestFitness = newValue;
                // }
            }
        }

        // System.out.println("找到的Pareto最优解:");
        // int number = 0;
        // for (double[] solution : paretoFront) {
        //     ++number;
        //     System.out.println("pareto_" + number
        //             + ": --目标函数1: " + solution[0]
        //             + ", 目标函数2: " + (-solution[1]));
        // }
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

    // 记录pareto解对应position数组
    private void addParetoPosition(double[] pos) {
        double[] newParetoPos = new double[numDimensions * 2];
        for (int i = 0; i < numDimensions * 2; i++) {
            newParetoPos[i] = pos[i];
        }

        paretoFrontPos.add(newParetoPos);
    }


    // 更新 pareto 解
    private void updateParetoFront(double[] position, double[] value) {
        List<double[]> toRemove = new ArrayList<>();
        List<double[]> toRemovePos = new ArrayList<>();

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
            }

            ++posIndex;
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
