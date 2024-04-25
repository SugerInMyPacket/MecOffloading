package resource_allocation;

import java.util.Random;

public class ParticleSwarmOptimization {

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

        // 构造函数
        public ParticleSwarmOptimizer(int numParticles, int numDimensions) {
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
                    // 根据问题设置合适的初始范围
                    particles[i].position[j] = Math.random() * 10;
                    // 根据问题设置合适的速度范围
                    particles[i].velocity[j] = Math.random() * 2 - 1;
                    particles[i].personalBest[j] = particles[i].position[j];
                }

                // 计算初始适应度
                double fitness = evaluateFitness(particles[i].position);

                // 更新个体最优适应度和位置
                particles[i].personalBestFitness = fitness;
                if (i == 0 || fitness < globalBestFitness) {
                    globalBestFitness = fitness;
                    globalBest = particles[i].position.clone();
                }
            }
        }

        // 评估适应度的方法，根据实际问题进行实现
        private double evaluateFitness(double[] position) {
            // 这里可以是需要优化的目标函数
            // 例如：return (position[0] - 2) * (position[0] - 2) + (position[1] - 3) * (position[1] - 3);
            return 0;
        }

        // 粒子群算法的主要迭代过程
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

    // 主函数
    public static void main(String[] args) {
        int numParticles = 30; // 粒子数量
        int numDimensions = 5; // 问题的维度，根据实际问题设置
        int numIterations = 100; // 迭代次数
        double inertiaWeight = 0.5; // 惯性权重
        double cognitiveWeight = 1.5; // 个体认知权重
        double socialWeight = 1.5; // 群体社会权重

        // 创建粒子群优化器
        ParticleSwarmOptimizer optimizer = new ParticleSwarmOptimizer(numParticles, numDimensions);

        // 执行粒子群算法
        optimizer.optimize(numIterations, inertiaWeight, cognitiveWeight, socialWeight);
    }
}
