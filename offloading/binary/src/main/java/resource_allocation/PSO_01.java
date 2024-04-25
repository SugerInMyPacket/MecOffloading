package resource_allocation;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class PSO_01 {

    // 定义粒子类
    static class Particle {
        double[] position; // 粒子当前位置
        double[] velocity; // 粒子当前速度

        double[] personalBest; // 粒子个体最优位置
        double personalBestFitness; // 粒子个体最优适应度
    }

    // 粒子群算法的主要类
    static class ParticleSwarmOptimizer {
        int numParticles; // 粒子数量
        double[] globalBest; // 群体最优位置
        double globalBestFitness; // 群体最优适应度
        Particle[] particles; // 粒子数组

        /**
         * 构造函数
         * @param numParticles  : 粒子数量
         * @param numDimensions  ： 维度
         * @param resourceArr  ： 资源分配量数组
         */
        public ParticleSwarmOptimizer(int numParticles, int numDimensions,
                                      List<Integer> resourceArr) {
            this.numParticles = numParticles;
            this.particles = new Particle[numParticles];

            // 初始化粒子
            for (int i = 0; i < numParticles; i++) {
                particles[i] = new Particle();
                particles[i].position = new double[numDimensions];
                particles[i].velocity = new double[numDimensions];
                particles[i].personalBest = new double[numDimensions];

                // 随机初始化粒子位置和速度
                for (int j = 0; j < numDimensions; j++) {
                    // 获取粒子当前的资源分配值(resourceArr[j])
                    particles[i].position[j] = resourceArr.get(j);
                    // TODO:设置速度范围
                    particles[i].velocity[j] = Math.random() * 2 - 1;
                    particles[i].personalBest[j] = particles[i].position[j];
                }

                // 计算初始适应度
                double fitness = evaluateFitness(particles[i].position);

                // 更新个体初始最优适应度和位置
                particles[i].personalBestFitness = fitness;
                if (i == 0 || fitness < globalBestFitness) {
                    globalBestFitness = fitness;
                    globalBest = particles[i].position.clone();
                }
            }
        }

        // TODO:适应度计算公式
        private double evaluateFitness(double[] position) {
            double sum = 0;
            for(double i : position) {
                sum += i;
            }
            double r = new Random().nextDouble();
            return sum * r;
            // return 0;
        }

        /**
         * 更新粒子
         * @param inertiaWeight   // 惯性权重
         * @param cognitiveWeight   // 个体认知权重
         * @param socialWeight   // 群体社会权重
         * @param minEdge   // 边界
         * @param maxEdge
         */
        public void optimize(double inertiaWeight, double cognitiveWeight, double socialWeight,
                             int minEdge, int maxEdge) {
            Random rand = new Random();
            // 更新每个粒子的位置和速度
            for (int i = 0; i < numParticles; i++) {
                for (int j = 0; j < globalBest.length; j++) {
                    // 更新速度
                    particles[i].velocity[j] = inertiaWeight * particles[i].velocity[j]
                            + cognitiveWeight * rand.nextDouble() * (particles[i].personalBest[j] - particles[i].position[j])
                            + socialWeight * rand.nextDouble() * (globalBest[j] - particles[i].position[j]);

                    // 更新位置
                    particles[i].position[j] += particles[i].velocity[j];

                    // TODO：防止位置越界，根据实际问题进行调整
                    if (particles[i].position[j] < minEdge) {
                        particles[i].position[j] = minEdge;
                    } else if (particles[i].position[j] > maxEdge) {
                        particles[i].position[j] = maxEdge;
                    }
                }

                // 计算新位置的适应度
                double newFitness = evaluateFitness(particles[i].position);

                // 更新个体最优适应度和位置
                if (newFitness < particles[i].personalBestFitness) {
                    particles[i].personalBestFitness = newFitness;
                    particles[i].personalBest = particles[i].position.clone();
                }

                // 更新群体最优适应度和位置
                if (newFitness < globalBestFitness) {
                    globalBestFitness = newFitness;
                    globalBest = particles[i].position.clone();
                }
            }

            // 输出每轮迭代后的最优解
            System.out.println(": Best Fitness = " + globalBestFitness
                    + ", Best Position = " + java.util.Arrays.toString(globalBest));
        }
        /**
         * 粒子群算法的主要迭代过程
         * @param numIterations  : 迭代次数
         * @param inertiaWeight
         * @param cognitiveWeight
         * @param socialWeight
         */
        public void optimize(int numIterations, double inertiaWeight,
                             double cognitiveWeight, double socialWeight) {
            Random rand = new Random();

            for (int iteration = 0; iteration < numIterations; iteration++) {
                // 更新每个粒子的位置和速度
                for (int i = 0; i < numParticles; i++) {
                    for (int j = 0; j < globalBest.length; j++) {
                        // 更新速度
                        particles[i].velocity[j] = inertiaWeight * particles[i].velocity[j]
                                + cognitiveWeight * rand.nextDouble() * (particles[i].personalBest[j] - particles[i].position[j])
                                + socialWeight * rand.nextDouble() * (globalBest[j] - particles[i].position[j]);

                        // 更新位置
                        particles[i].position[j] += particles[i].velocity[j];

                        // 防止位置越界，根据实际问题进行调整
                        // 可以根据问题的具体情况添加其他约束条件
                        if (particles[i].position[j] < 0) {
                            particles[i].position[j] = 0;
                        } else if (particles[i].position[j] > 10) {
                            particles[i].position[j] = 10;
                        }
                    }

                    // 计算新位置的适应度
                    double newFitness = evaluateFitness(particles[i].position);

                    // 更新个体最优适应度和位置
                    if (newFitness < particles[i].personalBestFitness) {
                        particles[i].personalBestFitness = newFitness;
                        particles[i].personalBest = particles[i].position.clone();
                    }

                    // 更新群体最优适应度和位置
                    if (newFitness < globalBestFitness) {
                        globalBestFitness = newFitness;
                        globalBest = particles[i].position.clone();
                    }
                }

                // 输出每轮迭代后的最优解
                System.out.println("Iteration " + (iteration + 1)
                        + ": Best Fitness = " + globalBestFitness
                        + ", Best Position = " + java.util.Arrays.toString(globalBest));
            }
        }
    }

    public static void main(String[] args) {
        int numParticles = 30; // 粒子数量
        int numDimensions = 5; // 问题的维度，根据实际问题设置
        int numIterations = 100; // 迭代次数
        double inertiaWeight = 0.5; // 惯性权重
        double cognitiveWeight = 1.5; // 个体认知权重
        double socialWeight = 1.5; // 群体社会权重

        List<Integer> resArr = new ArrayList<>();
        for(int i = 0; i < numDimensions; i++) {
            resArr.add(new Random().nextInt(5) + 1);
        }
        // 创建粒子群优化器
        ParticleSwarmOptimizer optimizer = new ParticleSwarmOptimizer(numParticles, numDimensions, resArr);

        // 执行粒子群算法
        for (int i = 0; i < numIterations; i++) {
            // optimizer.optimize(inertiaWeight, cognitiveWeight, socialWeight, 0, 10);
        }   
        optimizer.optimize(numIterations, inertiaWeight, cognitiveWeight, socialWeight);
    }
}
