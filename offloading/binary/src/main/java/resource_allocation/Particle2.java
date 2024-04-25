package resource_allocation;

import java.util.List;
import java.util.Random;

public class Particle2 {
    // 坐标
    double[] position;
    // 速度
    double[] velocity;
    // 个体最佳位置
    double[] pBestPosition;
    // 最佳适应度
    double[] pBestFitness;

    public Particle2(int dim, double[][] bounds, List<Integer> currUnloadArr) {
        Random random = new Random();

        position = new double[dim];
        velocity = new double[dim];
        pBestPosition = new double[dim];

        pBestFitness = new double[2]; // 两个目标值

        // 初始化
        for (int j = 0; j < dim; j++) {
            // position[j] = bounds[j][0] + random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            // velocity[j] = random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            position[j] = bounds[j][0] + random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            velocity[j] = random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            pBestPosition[j] = position[j];
        }

        pBestFitness = ObjFunction.evaluate(currUnloadArr, pBestPosition);
    }

}
