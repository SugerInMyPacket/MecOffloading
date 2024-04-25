package compare.psoco;

import utils.ArrayUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class PSOCO {

    int numParticles; // 粒子数量
    int numDimensions; // 维度数量  2*d
    int bound1; // unload 搜索空间边界
    double[] bounds2;
    double[][] bounds3; // freq 搜索空间边界
    double[][] bounds4;
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

    public PSOCO(int numParticles, int numDimensions,
                 int bound1, double[][] bounds3, double[][] bounds4,
                 double inertiaWeight, double cognitiveWeight, double socialWeight) {

        this.numParticles = numParticles;
        this.numDimensions = numDimensions;
        this.bound1 = bound1;
        this.bounds3 = bounds3;
        this.bounds4 = bounds4;
        this.inertiaWeight = inertiaWeight;
        this.cognitiveWeight = cognitiveWeight;
        this.socialWeight = socialWeight;
        this.paretoFront = new ArrayList<>();
        this.paretoFrontPos = new ArrayList<>();
        this.particles = new Particle_PSOCO[numParticles];
        this.random = new Random();

        for (int i = 0; i < numParticles; i++) {
            particles[i] = new Particle_PSOCO(numDimensions, bound1, bounds3, bounds4);
            // 更新 pareto 解
            updateParetoFront(particles[i].pBestPosition, particles[i].pBestFitness);
        }

    }

    public void optimize(int maxIterations) {

        // 代数
        for (int iter = 0; iter < maxIterations; iter++) {
            // 粒子
            for (int i = 0; i < numParticles; i++) {
                double[] newPosUnload = new double[numDimensions];
                double[] newPosUnloadRatio = new double[numDimensions];
                double[] newPosFreqLocal = new double[numDimensions];
                double[] newPosFreqRemote = new double[numDimensions];

                double[] newVelUnload = new double[numDimensions];
                double[] newVelUnloadRatio = new double[numDimensions];
                double[] newVelFreqLocal = new double[numDimensions];
                double[] newVelFreqRemote = new double[numDimensions];

                int sel_index = random.nextInt(paretoFront.size());
                double c1 = random.nextDouble();
                double c2 = random.nextDouble();

                // 更新惯性参数 w （inertiaWeight） [0.4,2]
                inertiaWeight = 2.0 - iter * (2.0 - 0.4) / maxIterations;

                // 维度
                for (int j = 0; j < numDimensions; j++) {
                    double cognitiveComponent = 0.0;
                    double socialComponent = 0.0;

                    int k = 0;
                    // 卸载决策
                    cognitiveComponent
                            = cognitiveWeight * c1 * (particles[i].pBestPosition[j] - particles[i].posUnload[j]);
                    socialComponent
                            = socialWeight * c2 * (paretoFrontPos.get(sel_index)[j] - particles[i].posUnload[j]);
                    newVelUnload[j]
                            = inertiaWeight * particles[i].velUnload[j] + cognitiveComponent + socialComponent;
                    newPosUnload[j]
                            = particles[i].posUnload[j] + newVelUnload[j];

                    // 卸载比率
                    cognitiveComponent
                            = cognitiveWeight * c1 * (particles[i].pBestPosition[numDimensions + j] - particles[i].posUnloadRatio[j]);
                    socialComponent
                            = socialWeight * c2 * (paretoFrontPos.get(sel_index)[numDimensions + j] - particles[i].posUnloadRatio[j]);
                    newVelUnloadRatio[j]
                            = inertiaWeight * particles[i].velUnloadRatio[j] + cognitiveComponent + socialComponent;
                    newPosUnloadRatio[j]
                            = particles[i].posUnloadRatio[j] + newVelUnloadRatio[j];

                    // Freq Local
                    cognitiveComponent
                            = cognitiveWeight * c1 * (particles[i].pBestPosition[2 * numDimensions + j] - particles[i].posFreqLocal[j]);
                    socialComponent
                            = socialWeight * c2 * (paretoFrontPos.get(sel_index)[2 * numDimensions + j] - particles[i].posFreqLocal[j]);
                    newVelFreqLocal[j]
                            = inertiaWeight * particles[i].velFreqLocal[j] + cognitiveComponent + socialComponent;
                    newPosFreqLocal[j]
                            = particles[i].posFreqLocal[j] + newVelFreqLocal[j];

                    // Freq Remote
                    cognitiveComponent
                            = cognitiveWeight * c1 * (particles[i].pBestPosition[3 * numDimensions + j] - particles[i].posFreqRemote[j]);
                    socialComponent
                            = socialWeight * c2 * (paretoFrontPos.get(sel_index)[3 * numDimensions + j] - particles[i].posFreqRemote[j]);
                    newVelFreqRemote[j]
                            = inertiaWeight * particles[i].velFreqRemote[j] + cognitiveComponent + socialComponent;
                    newPosFreqRemote[j]
                            = particles[i].posFreqRemote[j] + newVelFreqRemote[j];

                    if (newPosUnload[j] < -1) {
                        newPosUnload[j] = -1;
                    } else if (newPosUnload[j] > bound1) {
                        newPosUnload[j] = bound1;
                    }

                    if (newPosUnloadRatio[j] < 0.0) {
                        newPosUnloadRatio[j] = 0.0;
                    } else if (newPosUnloadRatio[j] > 1.0) {
                        newPosUnloadRatio[j] = 1.0;
                    }

                    if (newPosFreqLocal[j] < bounds3[j][0]) {
                        newPosFreqLocal[j] = bounds3[j][0];
                    } else if (newPosFreqLocal[j] > bounds3[j][1]) {
                        newPosFreqLocal[j] = bounds3[j][1];
                    }

                    if (newPosFreqRemote[j] < bounds4[j][0]) {
                        newPosFreqRemote[j] = bounds4[j][0];
                    } else if (newPosFreqRemote[j] > bounds4[j][1]) {
                        newPosFreqRemote[j] = bounds4[j][1];
                    }

                }  // dim

                // 适应度值评估
                double[] newValue
                        = ObjFunc_PSOCO.evaluate(newPosUnload, newPosUnloadRatio, newPosFreqLocal, newPosFreqRemote);

                // 更新个体最优解和Pareto前沿
                if (isParetoOptimal(newValue)) {
                    particles[i].pBestFitness = newValue;
                    particles[i].pBestPosition
                            = ArrayUtils.connectArrays(newPosUnload, newPosUnloadRatio, newPosFreqLocal, newPosFreqRemote);

                    // addParetoPosition(newPosition);
                    // 更新 pareto
                    updateParetoFront(particles[i].pBestPosition, newValue);
                }

                particles[i].velUnload = newVelUnload;
                particles[i].posUnload = newPosUnload;
                particles[i].velUnloadRatio = newVelUnloadRatio;
                particles[i].posUnloadRatio = newPosUnloadRatio;
                particles[i].velFreqLocal = newVelFreqLocal;
                particles[i].posFreqLocal = newPosFreqLocal;
                particles[i].velFreqRemote = newVelFreqRemote;
                particles[i].posFreqRemote = newPosFreqRemote;
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
