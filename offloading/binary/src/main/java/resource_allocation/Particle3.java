package resource_allocation;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Particle3 {
    // 坐标
    double[] position;
    // 速度
    double[] velocity;
    // 个体最佳位置
    double[] pBestPosition;
    // 最佳适应度
    double[] pBestFitness;

    int[] currParticleUnloadArr;
    List<Integer> currParticleUnloadList;

    public Particle3(int dim, double[][] bounds, List<Integer> currUnloadArr) {
        Random random = new Random();

        position = new double[dim];
        velocity = new double[dim];
        pBestPosition = new double[dim];

        currParticleUnloadArr = new int[dim];
        currParticleUnloadList = new ArrayList<>();

        pBestFitness = new double[2]; // 两个目标值

        // 初始化
        for (int j = 0; j < dim; j++) {
            currParticleUnloadArr[j] = currUnloadArr.get(j);
            currParticleUnloadList.add(currUnloadArr.get(j));
            // position[j] = bounds[j][0] + random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            // velocity[j] = random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            position[j] = bounds[j][0] + random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            velocity[j] = random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            pBestPosition[j] = position[j];
        }

        // pBestFitness = ObjFunction.evaluate(currUnloadArr, pBestPosition);
        pBestFitness = ObjFuncPSO2.evaluate(currParticleUnloadList, pBestPosition);
    }

    public Particle3(int dim, double[][] bounds,
                     List<Integer> currUnloadArr,
                     List<Integer> currInputFreqArr) {
        Random random = new Random();

        position = new double[dim];
        velocity = new double[dim];
        pBestPosition = new double[dim];

        currParticleUnloadArr = new int[dim];
        currParticleUnloadList = new ArrayList<>();

        pBestFitness = new double[2]; // 两个目标值

        // 初始化
        for (int j = 0; j < dim; j++) {
            currParticleUnloadArr[j] = currUnloadArr.get(j);
            currParticleUnloadList.add(currUnloadArr.get(j));
            // position[j] = bounds[j][0] + random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            // velocity[j] = random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            position[j] = bounds[j][0] + random.nextDouble() * currInputFreqArr.get(j);
            velocity[j] = random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            pBestPosition[j] = position[j];
        }

        // pBestFitness = ObjFunction.evaluate(currUnloadArr, pBestPosition);
        pBestFitness = ObjFuncPSO2.evaluate(currParticleUnloadList, pBestPosition);
    }

}
