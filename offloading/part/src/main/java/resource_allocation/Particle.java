package resource_allocation;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Particle {
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
    List<Double> currParticleUnloadRatioList;

    public Particle(int dim,
                    double[][] bounds,
                    List<Integer> currUnloadArr,
                    List<Double> currUnloadRatioArr) {
        Random random = new Random();

        position = new double[dim];
        velocity = new double[dim];
        pBestPosition = new double[dim];

        currParticleUnloadArr = new int[dim];
        currParticleUnloadList = new ArrayList<>();
        currParticleUnloadRatioList = new ArrayList<>();

        pBestFitness = new double[2]; // 两个目标值

        // 初始化
        for (int j = 0; j < dim / 2; j++) {
            // currParticleUnloadArr[j] = currUnloadArr.get(j);
            currParticleUnloadList.add(currUnloadArr.get(j));
            currParticleUnloadRatioList.add(currUnloadRatioArr.get(j));
        }
        for (int j = 0; j < dim; j++) {
            // currParticleUnloadArr[j] = currUnloadArr.get(j);
            // currParticleUnloadList.add(currUnloadArr.get(j));
            // currParticleUnloadRatioList.add(currUnloadRatioArr.get(j));
            // position[j] = bounds[j][0] + random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            // velocity[j] = random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            position[j] = bounds[j][0] + random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            velocity[j] = random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            pBestPosition[j] = position[j];
        }

        // pBestFitness = ObjFunction.evaluate(currUnloadArr, pBestPosition);
        pBestFitness = ObjFuncPSO.evaluate(currParticleUnloadList,
                currParticleUnloadRatioList, pBestPosition);
    }

    public Particle(int dim,
                    double[][] bounds,
                    List<Integer> currUnloadArr,
                    List<Double> currUnloadRatioArr,
                    List<Integer> currInputFreqArrLocal,
                    List<Integer> currInputFreqArrRemote) {
        Random random = new Random();

        position = new double[dim];
        velocity = new double[dim];
        pBestPosition = new double[dim];

        currParticleUnloadArr = new int[dim];
        currParticleUnloadList = new ArrayList<>();
        currParticleUnloadRatioList = new ArrayList<>();

        pBestFitness = new double[2]; // 两个目标值

        // 初始化
        for (int j = 0; j < dim; j++) {
            currParticleUnloadArr[j] = currUnloadArr.get(j);
            currParticleUnloadList.add(currUnloadArr.get(j));
            currParticleUnloadRatioList.add(currUnloadRatioArr.get(j));
            // position[j] = bounds[j][0] + random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            // velocity[j] = random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            position[j] = bounds[j][0] + random.nextDouble() * currInputFreqArrLocal.get(j);
            velocity[j] = random.nextDouble() * (bounds[j][1] - bounds[j][0]);
            pBestPosition[j] = position[j];
        }

        // pBestFitness = ObjFunction.evaluate(currUnloadArr, pBestPosition);
        pBestFitness = ObjFuncPSO.evaluate(currParticleUnloadList,
                currParticleUnloadRatioList, pBestPosition);
    }

}
