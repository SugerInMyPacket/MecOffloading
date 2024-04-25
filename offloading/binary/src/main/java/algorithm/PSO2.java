package algorithm;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class PSO2 {

    static Random random = new Random();
    static int populationSize = 10; // 粒子数量
    static double inertiaWeight = 0.5; // 惯性权重
    static double cognitiveWeight = 0.5; // 学习因子权重
    static double socialWeight = 0.5; // 社会因子权重

    // 粒子群算法更新后k位
    public static List<Integer> updateParticle(List<Integer> list, int k, int n) {
        // 实现粒子群算法更新后k位的逻辑
        List<Particle> particles = generateParticles(k, n);

        // 迭代次数，可根据需要修改
        for (int iteration = 0; iteration < 100; iteration++) {
            for (Particle particle : particles) {
                List<Integer> newPosition = updateParticlePosition(particle, list, k, n);
                particle.setPosition(newPosition);
            }
        }

        // 选择最优粒子
        Particle bestParticle = particles.get(0);
        for (Particle particle : particles) {
            if (particle.getFitness() > bestParticle.getFitness()) {
                bestParticle = particle;
            }
        }

        List<Integer> bestPosition = bestParticle.getPosition();
        for (int i = k; i < 2 * k; i++) {
            list.set(i, bestPosition.get(i - k)); // 更新后k位的值
        }

        return list.subList(k, 2 * k); // 返回更新后的后k位列表
    }

    // 生成初始粒子群
    private static List<Particle> generateParticles(int k, int n) {
        List<Particle> particles = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            List<Integer> position = new ArrayList<>();
            for (int j = 0; j < k; j++) {
                position.add(random.nextInt(n));
            }
            Particle particle = new Particle(position);
            particles.add(particle);
        }
        return particles;
    }

    // 更新粒子位置
    private static List<Integer> updateParticlePosition(Particle particle, List<Integer> list, int k, int n) {
        List<Integer> currentPosition = particle.getPosition();
        List<Integer> currentVelocity = particle.getVelocity();
        List<Integer> newVelocity = new ArrayList<>();
        List<Integer> newPosition = new ArrayList<>();

        // 更新粒子速度和位置
        for (int i = 0; i < k; i++) {
            int inertia = (int) (inertiaWeight * currentVelocity.get(i));
            int cognitive = random.nextInt((int) (cognitiveWeight * (currentPosition.get(i) - list.get(i))));
            int social = random.nextInt((int) (socialWeight * (currentPosition.get(i) - list.get(i))));
            int velocity = inertia + cognitive + social;

            newVelocity.add(velocity);
            int positionValue = currentPosition.get(i) + velocity;

            // 控制粒子位置在合理范围内
            if (positionValue < 0) {
                positionValue = 0;
            } else if (positionValue > n) {
                positionValue = n;
            }
            newPosition.add(positionValue);
        }

        particle.setVelocity(newVelocity);
        return newPosition;
    }

    // 粒子类
    static class Particle {
        private List<Integer> position;
        private List<Integer> velocity;

        public Particle(List<Integer> position) {
            this.position = position;
            this.velocity = new ArrayList<>();
            for (int i = 0; i < position.size(); i++) {
                this.velocity.add(0);
            }
        }

        public List<Integer> getPosition() {
            return position;
        }

        public void setPosition(List<Integer> position) {
            this.position = position;
        }

        public List<Integer> getVelocity() {
            return velocity;
        }

        public void setVelocity(List<Integer> velocity) {
            this.velocity = velocity;
        }

        // 计算粒子适应度
        public double getFitness() {
            // 这里简单地以粒子位置的和作为适应度函数，你可以根据实际情况设计更合适的适应度函数
            double sum = 0;
            for (int value : position) {
                sum += value;
            }
            return sum;
        }
    }

}
