import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ParticleSwarmOptimization {

    // 粒子定义
    static class Particle {
        double[] position;
        double[] velocity;
        double[] pBestPosition;
        double[] pBestValue;

        public Particle(int dim) {
            position = new double[dim];
            velocity = new double[dim];
            pBestPosition = new double[dim];
            pBestValue = new double[2]; // 两个目标值
        }
    }

    // 目标函数
    static class ObjectiveFunction {
        // 根据你的目标函数进行修改
        static double[] evaluate(double[] x) {
            double[] result = new double[2];
            result[0] = x[0] * x[0]; // 目标函数1
            result[1] = (x[0] - 2) * (x[0] - 2) + x[1] * x[1]; // 目标函数2
            return result;
        }
    }

    // PSO算法
    static class PSO {
        int numParticles; // 粒子数量
        int numDimensions; // 维度数量
        double[][] bounds; // 搜索空间边界
        double inertiaWeight; // 惯性权重
        double cognitiveWeight; // 认知权重
        double socialWeight; // 社会权重
        List<double[]> paretoFront; // Pareto前沿
        Particle[] particles; // 粒子群
        Random random;

        // 初始化构造
        public PSO(int numParticles, int numDimensions, double[][] bounds,
                   double inertiaWeight, double cognitiveWeight, double socialWeight) {
            this.numParticles = numParticles;
            this.numDimensions = numDimensions;
            this.bounds = bounds;
            this.inertiaWeight = inertiaWeight;
            this.cognitiveWeight = cognitiveWeight;
            this.socialWeight = socialWeight;
            this.paretoFront = new ArrayList<>();
            this.particles = new Particle[numParticles];
            this.random = new Random();

            for (int i = 0; i < numParticles; i++) {
                particles[i] = new Particle(numDimensions);
                for (int j = 0; j < numDimensions; j++) {
                    particles[i].position[j] = bounds[j][0] + random.nextDouble() * (bounds[j][1] - bounds[j][0]);
                    particles[i].velocity[j] = random.nextDouble() * (bounds[j][1] - bounds[j][0]);
                    particles[i].pBestPosition[j] = particles[i].position[j];
                }
                particles[i].pBestValue = ObjectiveFunction.evaluate(particles[i].pBestPosition);
                updateParetoFront(particles[i].pBestPosition, particles[i].pBestValue);
            }
        }

        // 优化迭代
        public void optimize(int maxIterations) {
            for (int iter = 0; iter < maxIterations; iter++) {
                for (int i = 0; i < numParticles; i++) {
                    double[] newVelocity = new double[numDimensions];
                    double[] newPosition = new double[numDimensions];

                    for (int j = 0; j < numDimensions; j++) {
                        // 计算认知和社会分量
                        double cognitiveComponent = cognitiveWeight * random.nextDouble() * (particles[i].pBestPosition[j] - particles[i].position[j]);
                        double socialComponent = socialWeight * random.nextDouble() * (paretoFront.get(random.nextInt(paretoFront.size()))[j] - particles[i].position[j]);
                        // 更新速度和位置
                        newVelocity[j] = inertiaWeight * particles[i].velocity[j] + cognitiveComponent + socialComponent;
                        newPosition[j] = particles[i].position[j] + newVelocity[j];

                        // 确保新位置在边界内
                        if (newPosition[j] < bounds[j][0]) {
                            newPosition[j] = bounds[j][0];
                        } else if (newPosition[j] > bounds[j][1]) {
                            newPosition[j] = bounds[j][1];
                        }
                    }

                    double[] newValue = ObjectiveFunction.evaluate(newPosition);

                    // 更新个体最优解和Pareto前沿
                    if (isParetoOptimal(newValue)) {
                        particles[i].pBestValue = newValue;
                        particles[i].pBestPosition = newPosition;
                        updateParetoFront(newPosition, newValue);
                    }

                    particles[i].velocity = newVelocity;
                    particles[i].position = newPosition;
                }
            }

            System.out.println("找到的Pareto最优解:");
            for (double[] solution : paretoFront) {
                System.out.println("目标函数1: " + solution[0] + ", 目标函数2: " + solution[1]);
            }
        }

        // 双目标 --> pareto 定义
        private boolean isParetoOptimal(double[] newValue) {
            for (double[] solution : paretoFront) {
                // note：最小化问题
                // 如果newValue对于两个目标函数的解都大于已有解，则不是pareto解
                if (newValue[0] >= solution[0] && newValue[1] >= solution[1]) {
                    return false;
                }
            }
            return true;
        }

        private void updateParetoFront(double[] position, double[] value) {
            List<double[]> toRemove = new ArrayList<>();
            for (double[] solution : paretoFront) {
                if (solution[0] <= value[0] && solution[1] <= value[1]) {
                    return; // 新解支配已知解，不更新Pareto前沿
                } else if (solution[0] >= value[0] && solution[1] >= value[1]) {
                    toRemove.add(solution); // 已知解支配新解，需要移除
                }
            }
            paretoFront.add(value);
            paretoFront.removeAll(toRemove);
        }
    }

    public static void main(String[] args) {
        int numParticles = 20;
        int numDimensions = 2;
        double[][] bounds = {{-10, 10}, {-10, 10}};
        double inertiaWeight = 0.7;
        double cognitiveWeight = 1.5;
        double socialWeight = 1.5;
        int maxIterations = 100;

        PSO pso = new PSO(numParticles, numDimensions, bounds, inertiaWeight, cognitiveWeight, socialWeight);
        pso.optimize(maxIterations);
    }
}
