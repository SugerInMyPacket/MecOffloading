package pso;

import java.util.Random;

public class Particle {
    double[] position;
    double[] velocity;
    double[] pBestPosition;
    double[] pBestValue;
    
    public Particle(int dim, double[][] bounds, Random random) {
        position = new double[dim];
        velocity = new double[dim];
        pBestPosition = new double[dim];

        pBestValue = new double[2]; // 两个目标值

        // 初始化
        for (int j = 0; j < dim; j++) {
            position[j] = bounds[j][0] + random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            velocity[j] = random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            pBestPosition[j] = position[j];
        }

        pBestValue = ObjectiveFunction.evaluate(pBestPosition);
    }
}
