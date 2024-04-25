package unload_ratio;

import utils.NumUtil;

import java.util.List;
import java.util.Random;

public class Population2 {
    int[] solutionUnload;
    double[] solutionRatio;

    double[] solutionAll;

    double[] fitness;

    public Population2(){}

    public Population2(int dimensions,
                       int bound,
                       List<Integer> currFreqAllocArrLocal,
                       List<Integer> currFreqAllocArrRemote) {

        Random random = new Random();

        solutionUnload = new int[dimensions];
        solutionRatio = new double[dimensions];

        solutionAll = new double[dimensions * 2];

        // 双目标优化
        fitness = new double[2];

        // 初始化
        for (int i = 0; i < dimensions; i++) {
            solutionUnload[i] = random.nextInt(bound + 1);
            solutionRatio[i] = NumUtil.random(0.0, 1.0);

            solutionAll[i] = solutionUnload[i];
            solutionAll[i + dimensions] = solutionRatio[i];
        }

        // 适应度评估
        fitness = ObjFuncDE2.evaluate(solutionUnload, solutionRatio,
                currFreqAllocArrLocal, currFreqAllocArrRemote);

    }

    public Population2(int dimensions,
                       int bound,
                       List<Integer> currUnloadArr,
                       List<Double> currRatioArr,
                       List<Integer> currFreqAllocArrLocal,
                       List<Integer> currFreqAllocArrRemote) {

        Random random = new Random();

        solutionUnload = new int[dimensions];
        solutionRatio = new double[dimensions];
        solutionAll = new double[dimensions * 2];


        // 双目标优化
        fitness = new double[2];

        // 初始化
        for (int i = 0; i < dimensions; i++) {
            // solution[i] = random.nextInt(bound + 1);
            solutionUnload[i] = currUnloadArr.get(i);
            solutionRatio[i] = currRatioArr.get(i);

            solutionAll[i] = currUnloadArr.get(i);
            solutionAll[i + dimensions] = currRatioArr.get(i);

            // solutionAll[i] = solutionUnload[i];
            // solutionAll[i + dimensions] = solutionRatio[i];
        }

        // 适应度评估
        fitness = ObjFuncDE2.evaluate(solutionUnload, solutionRatio,
                currFreqAllocArrLocal, currFreqAllocArrRemote);

    }

}
