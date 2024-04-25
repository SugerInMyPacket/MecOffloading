package resource_allocation;


import utils.FormatData;

import java.util.List;
import java.util.Random;

public class PSO_02 {
    int numParticles; // 粒子数量
    double[] globalBestPos; // 群体最优位置
    double globalBestFitness; // 群体最优适应度
    // double[][] gBestPosK; // 记录 k 个群体最优位置
    // double[] globalBestFitnessK; // k 个群体最优适应度
    Particle[] particles; // 粒子数组

    Particle bestParticle;

    public PSO_02() {}

    public PSO_02(int numParticles, int numDimensions,
                  List<Integer> unloadArr, List<Integer> freqAllocArr) {

    }

    public Particle setGlobalBestParticle(Particle obj, Particle curr) {
        obj.pos = curr.pos.clone();
        return obj;
    }

    /**
     * 初始化粒子 ==> 根据聚类方案 + 一定随机性
     *
     * @param numParticles
     * @param numDimensions
     * @param resAllocArr
     */
    public void initParticles(int numParticles, int numDimensions, List<Integer> resAllocArr) {
        // numDimensions == tasks.size() == resAlloArr.size()

        this.numParticles = numParticles;
        particles = new Particle[numParticles];

        bestParticle = new Particle();  // 最优粒子

        // 初始化每个粒子相关属性
        for (int i = 0; i < numParticles; i++) {
            particles[i] = new Particle();
            particles[i].pos = new double[numDimensions];
            particles[i].vel = new double[numDimensions];
            particles[i].pBestPos = new double[numDimensions];

            // 初始化粒子位置和速度
            for (int j = 0; j < numDimensions; j++) {
                // 获取粒子当前的资源分配值(resourceArr[j])
                particles[i].pos[j] = resAllocArr.get(j);
                // TODO:设置速度范围
                particles[i].vel[j] = Math.random() * 2 - 1;
                // 初始化 个体最优位置
                particles[i].pBestPos[j] = particles[i].pos[j];
            }

            // 计算粒子适应度
            double currFitness = evaluateFitness(particles[i]);

            // 更新个体初始适应度
            particles[i].pBestFitness = currFitness;
            // 更新种群最优适应度和位置
            // note: > or <
            if (i == 0 || currFitness < globalBestFitness) {
                globalBestFitness = currFitness;
                globalBestPos = particles[i].pos.clone();
            }
        }

        bestParticle.pBestFitness = globalBestFitness;
        bestParticle.pos = globalBestPos.clone();
    }

    /**
     * note：初始化粒子 ==> 根据上一轮迭代的最优粒子
     *
     * @param numParticles
     * @param numDimensions
     * @param oldBestParticle
     */
    public void initParticles2(int numParticles, int numDimensions, List<Integer> oldBestParticle) {

    }

    // TODO: 粒子适应度计算
    // question：选择 整个粒子 卸载决策+资源分配 ==> 适应度 还是 仅考虑 资源分配
    public double evaluateFitness(Particle particle) {
        // 应该需要传入 lamdaVar ==>
        double[] currParticlePos = particle.pos;
        // 根据pos，适应度计算公式
        return FormatData.getEffectiveValue4Digit(new Random().nextDouble(), 5);
        // return 0;
    }

    /**
     * 迭代优化粒子
     *
     * @param numIterations   ：迭代次数
     * @param inertiaWeight   ： 惯性
     * @param cognitiveWeight ： 学习因子
     * @param socialWeight
     * @param minPos          ： 位置边界
     * @param maxPos
     * @param minVel          ： 速度边界
     * @param maxVel
     *
     * note：根据迭代次数前后期，改变权重的变化
     */
    public void optimizeParticles(int numIterations, double inertiaWeight,
                                  double cognitiveWeight, double socialWeight,
                                  int minPos, int maxPos,
                                  int minVel, int maxVel) {
        // 声明随机数
        Random r = new Random();

        for (int iter = 0; iter < numIterations; iter++) {
            // 更新每个粒子的 pos 和 vel
            for (int i = 0; i < numParticles; i++) {
                for (int j = 0; j < globalBestPos.length; j++) {
                    // update vel
                    particles[i].vel[j] =
                            inertiaWeight * particles[i].vel[j]
                                    + cognitiveWeight * r.nextDouble() * (particles[i].pBestPos[j] - particles[i].pos[j])
                                    + socialWeight * r.nextDouble() * (globalBestPos[j] - particles[i].pos[j]);

                    // TODO: 速度界限限制

                    // update pos
                    particles[i].pos[j] += particles[i].vel[j];

                    // TODO: 位置越界条件判断
                    if (particles[i].pos[j] < minPos) {
                        // ...
                    } else if (particles[i].pos[j] > maxPos) {
                        // ...
                    }
                }

                // 计算粒子更新后的适应度
                double newFitness = evaluateFitness(particles[i]);

                // 更新个体最优适应度和最优位置
                if (newFitness < particles[i].pBestFitness) {
                    // note: > or <
                    particles[i].pBestFitness = newFitness;
                    particles[i].pBestPos = particles[i].pos.clone();
                }

                // 更新种群最优适应度和最优位置
                // note: 多个 globalBestFitness 时，轮盘赌策略 --> 替换
                if (newFitness < globalBestFitness) {
                    globalBestFitness = newFitness;
                    globalBestPos = particles[i].pos.clone();
                }
            }

            // setGlobalBestParticle(obj, globalBestPos);
            // 输出每轮迭代后的最优解
            System.out.println("ite:" + (iter + 1)
                    + ": Best Fitness = " + globalBestFitness
                    + ": best pos = " + java.util.Arrays.toString(globalBestPos));
        }

        bestParticle.pBestFitness = globalBestFitness;
        bestParticle.pos = globalBestPos.clone();

    }

    // 得到当前最优的个体
    public Particle getBestParticle() {
        return bestParticle;
    }

    // note：根据迭代次数 currIterations 与 总迭代次数，改变权重的变化
    public void optimizeParticles(int numIterations, int currIterations,
                                  double inertiaWeight, double cognitiveWeight, double socialWeight,
                                  int minEdge, int maxEdge) {

    }

    public void optimizeParticles(double inertiaWeight,
                                  double cognitiveWeight, double socialWeight,
                                  int minEdge, int maxEdge) {

    }
}
