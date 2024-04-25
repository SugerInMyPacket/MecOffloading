package compare;

import resource_allocation.ObjFunction;

import java.util.List;
import java.util.Random;

public class Particle_PSOCO {
    // 坐标
    double[] position;
    // 速度
    double[] velocity;
    // 个体最佳位置
    double[] pBestPosition;
    // 最佳适应度
    double[] pBestFitness;

    public Particle_PSOCO(int dim, int bound1, double[][] bounds2) {
        Random random = new Random();

        position = new double[dim * 2];
        velocity = new double[dim * 2];
        pBestPosition = new double[dim * 2];
        pBestFitness = new double[2]; // 两个目标值


        int[] unloadArr = new int[dim];
        double[] freqArr = new double[dim];

        // 初始化
        for (int j = 0; j < dim; j++) {
            // unload
            position[j] = -1 + random.nextDouble() * (bound1 + 1);
            velocity[j] = -1 + random.nextDouble() * (bound1 + 1);
            // freq
            position[j + dim] = bounds2[j][0] + random.nextDouble() * (bounds2[j][1] - bounds2[j][0]);
            velocity[j + dim] = random.nextDouble() * (bounds2[j][1] - bounds2[j][0]);

            pBestPosition[j] = position[j];
            pBestPosition[j + dim] = position[j + dim];
        }

        for (int j = 0; j < dim; j++) {
            unloadArr[j] = (int) pBestPosition[j];
            freqArr[j] = pBestPosition[j + dim];
        }

        pBestFitness = ObjFunc_PSOCO.evaluate(unloadArr, freqArr);

        // 计算 Fitness 后，unloadArr 会被修改，将修正后的值返回给 pBestPosition
        for (int j = 0; j < dim; j++) {
            position[j] = unloadArr[j];
            position[j + dim] = freqArr[j];
            pBestPosition[j] = unloadArr[j];
            pBestPosition[j + dim] = freqArr[j];
        }

    }
}
