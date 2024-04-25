package unload_ratio;

import config.InitFrame;
import entity.Cloud;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import enums.AlgorithmParam;
import enums.TaskPolicy;
import lombok.extern.slf4j.Slf4j;
import resource_allocation.PSO;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

@Slf4j(topic = "Test_DPRCO")
public class TestDPRCO {

    static List<Task> taskList = new ArrayList<>();

    static RoadsideUnit rsu = new RoadsideUnit();

    static Cloud cloud = new Cloud();

    // 车辆数目
    static int vehicleNums = 5;
    static List<Vehicle> vehicleList = new ArrayList<>();


    static int len = 10;

    static List<Integer> unloadArr = new ArrayList<>();
    static List<Double> unloadRatioArr = new ArrayList<>();
    static List<Integer> freqAllocArrLocal = new ArrayList<>();
    static List<Integer> freqAllocArrRemote = new ArrayList<>();

    static int taskSizeFromDB;
    static int vehicleSizeFromDB;


    public static void main(String[] args) {
        testDPRCO();

    }

    public static void testDPRCO() {

        log.info("\n");
        log.error("======== test DSPRCOA =========");

        // 初始化
        InitFrame.initFromDB();

        // 读取信息
        taskList = InitFrame.getTaskList();
        vehicleList = InitFrame.getVehicleList();
        rsu = InitFrame.getRSU();
        cloud = InitFrame.getCloud();

        taskSizeFromDB = taskList.size();
        vehicleSizeFromDB = vehicleList.size();
        log.info("taskSizeFromDB: " + taskSizeFromDB);
        log.info("vehicleSizeFromDB: " + vehicleSizeFromDB);

        // 根据任务类别，初始化资源分配
        List<Integer> unloadArrInit = new ArrayList<>(); // 个数应该为 geneLength
        List<Double> unloadRatioArrInit = new ArrayList<>();
        List<Integer> freqAllocArrLocalInit = new ArrayList<>();
        List<Integer> freqAllocArrRemoteInit = new ArrayList<>();

        for (int i = 0; i < taskSizeFromDB; i++) {
            unloadArrInit.add(taskList.get(i).getVehicleID());
            unloadRatioArrInit.add(0.5);
            // 根据 task 的 cluster_class 参数初始化
            int currTaskClusterID = taskList.get(i).getClusterID();  // 任务聚类 -> class_
            Random random = new Random();
            double ratio = random.nextDouble();
            int tempFreqDefault = (int) (ratio * TaskPolicy.TASK_FREQ_INIT_CLASS_DEFAULT);
            int tempFreqClass = 0;
            if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_0) {
                tempFreqClass = (int) ((1.0 - ratio) * TaskPolicy.TASK_FREQ_INIT_CLASS_0);
                freqAllocArrLocalInit.add(tempFreqDefault + tempFreqClass);
                freqAllocArrRemoteInit.add(tempFreqDefault + tempFreqClass);
            } else if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_1) {
                tempFreqClass = (int) ((1.0 - ratio) * TaskPolicy.TASK_FREQ_INIT_CLASS_1);
                freqAllocArrLocalInit.add(tempFreqDefault + tempFreqClass);
                freqAllocArrRemoteInit.add(tempFreqDefault + tempFreqClass);
            } else if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_2) {
                tempFreqClass = (int) ((1.0 - ratio) * TaskPolicy.TASK_FREQ_INIT_CLASS_2);
                freqAllocArrLocalInit.add(tempFreqDefault + tempFreqClass);
                freqAllocArrRemoteInit.add(tempFreqDefault + tempFreqClass);
            } else if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_3) {
                tempFreqClass = (int) ((1.0 - ratio) * TaskPolicy.TASK_FREQ_INIT_CLASS_3);
                freqAllocArrLocalInit.add(tempFreqDefault + tempFreqClass);
                freqAllocArrRemoteInit.add(tempFreqDefault + tempFreqClass);
            } else {
                freqAllocArrLocalInit.add(tempFreqDefault + tempFreqClass);
                freqAllocArrRemoteInit.add(tempFreqDefault + tempFreqClass);
            }

            // freqAllocArr.add(freqAllocArrInit.get(i));
        }

        int[] freqRemain = new int[vehicleSizeFromDB + 2];
        freqRemain[0] = (int) cloud.getFreqRemain();
        freqRemain[1] = (int) rsu.getFreqRemain();
        for (int i = 0; i < vehicleSizeFromDB; i++) {
            freqRemain[i + 2] = (int) vehicleList.get(i).getFreqRemain();
        }
        log.info("Init Freq Remain: \n" + Arrays.toString(freqRemain));

        Random random = new Random();


        // Round start ...
        List<int[]> paretoUnloadList = new ArrayList<>();
        List<double[]> paretoUnloadRatioList = new ArrayList<>();
        List<double[]> paretoFrontList = new ArrayList<>();
        List<double[]> paretoFrontFreqList = new ArrayList<>();
        // 大循环迭代次数
        int maxRound = AlgorithmParam.NUM_ROUND_TIMES;
        int numDimensions = taskSizeFromDB;
        int maxBound = vehicleSizeFromDB;


        // DE 参数
        int populationSize = 100;  // 个体数量
        int maxGenerations = 100;  // 最大迭代次数
        double crossoverRate = 0.8;  // 交叉概率
        double mutationFactor = 0.5;  // 变异概率

        // PSO 参数
        int numParticles = 100; // 粒子数量
        int numIterations = 100; // 迭代次数
        double inertiaWeight = 0.5; // 惯性权重
        double cognitiveWeight = 1.5; // 个体认知权重
        double socialWeight = 1.5; // 群体社会权重
        double[][] bounds = new double[numDimensions * 2][2];  // 边界
        for (int i = 0; i < numDimensions * 2; i++) {
            bounds[i][0] = 100;
            bounds[i][1] = 500;
        }


        for (int r = 0; r < maxRound; r++) {
            log.info("--- 第" + (r + 1) + "次大迭代 ---");

            DE_11 de = new DE_11(populationSize, maxGenerations, crossoverRate,
                    mutationFactor, numDimensions, maxBound,
                    unloadArrInit, unloadRatioArrInit,
                    freqAllocArrLocalInit, freqAllocArrRemoteInit);
            de.optimize();
            int paretoSize = 0;
            paretoSize = de.getParetoFront().size();
            log.info("DE - pareto 个数：" + paretoSize);
            // 随机选择一个（轮盘赌）
            int selIndex = random.nextInt(paretoSize);
            double[] solutionSelect = de.getParetoFrontSolution().get(selIndex);

            unloadArr.clear();
            for (int i = 0; i < numDimensions; i++) {
                unloadArr.add((int) solutionSelect[i]);
                unloadRatioArr.add(solutionSelect[i + numDimensions]);
            }

            // 重新定义粒子位置边界
            for (int i = 0; i < numDimensions; i++) {
                bounds[i][0] = AlgorithmParam.MIN_POS_PARTICLE;
                // bounds[i][1] = 500;
                bounds[i][1] = Math.min(AlgorithmParam.MAX_POS_PARTICLE, freqRemain[unloadArr.get(i) + 1]);
            }

            PSO pso = new PSO(numParticles, numDimensions * 2, bounds,
                    inertiaWeight, cognitiveWeight, socialWeight,
                    unloadArr, unloadRatioArr);
            pso.optimize(numIterations);

            paretoSize = pso.getParetoFront().size();
            log.info("PSO - pareto 个数：" + paretoSize);
            selIndex = random.nextInt(paretoSize);
            double[] freqArr = pso.getParetoFrontPos().get(selIndex);
            freqAllocArrLocalInit.clear();
            freqAllocArrRemoteInit.clear();
            for (int i = 0; i < numDimensions; i++) {
                freqAllocArrLocalInit.add((int) freqArr[i]);
                freqAllocArrRemoteInit.add((int) freqArr[i + numDimensions]);
            }

            // 记录迭代结束时的值
            if (r == maxRound - 1) {
                paretoUnloadList = pso.getParetoUnloadArr();
                paretoUnloadRatioList = pso.getParetoUnloadRatioArr();
                paretoFrontList = pso.getParetoFront();
                paretoFrontFreqList = pso.getParetoFrontPos();
            }

        }
        log.warn("★★★★★ 找到的 Pareto 最优解: ★★★★★");
        for (double[] solution : paretoFrontList) {
            log.info("USS avg val: " + solution[0] + ", Energy avg val: " + (-solution[1]));
        }

        log.info(Arrays.toString(paretoUnloadList.get(0)));
        log.info(Arrays.toString(paretoUnloadRatioList.get(0)));


    }
}
