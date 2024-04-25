package unload_decision;

import java.util.List;
import java.util.Random;

public class Population {
    int[] solution;

    double[] fitness;

    public Population(){}

    public Population(int dimensions,
                      int bound,
                      List<Double> currUnloadRatioArr,
                      List<Integer> currFreqAllocArrLocal,
                      List<Integer> currFreqAllocArrRemote) {

        Random random = new Random();

        solution = new int[dimensions];

        // 双目标优化
        fitness = new double[2];

        // 初始化
        for (int i = 0; i < dimensions; i++) {
            solution[i] = random.nextInt(bound + 1);
        }

        // 适应度评估
        fitness = ObjFuncDE.evaluate(solution, currUnloadRatioArr,
                currFreqAllocArrLocal, currFreqAllocArrRemote);

    }

    public Population(int dimensions,
                      int bound,
                      List<Integer> currUnloadArr,
                      List<Double> currUnloadRatioArr,
                      List<Integer> currFreqAllocArrLocal,
                      List<Integer> currFreqAllocArrRemote) {

        Random random = new Random();

        solution = new int[dimensions];

        // 双目标优化
        fitness = new double[2];

        // 初始化
        for (int i = 0; i < dimensions; i++) {
            // solution[i] = random.nextInt(bound + 1);
            solution[i] = currUnloadArr.get(i);
        }

        // 适应度评估
        fitness = ObjFuncDE.evaluate(solution, currUnloadRatioArr,
                currFreqAllocArrLocal, currFreqAllocArrRemote);

    }

}
