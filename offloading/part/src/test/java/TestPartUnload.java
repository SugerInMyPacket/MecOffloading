import config.InitFrame;
import entity.Cloud;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import enums.AlgorithmParam;
import enums.TaskPolicy;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import ratio_division.SA;
import resource_allocation.PSO;
import unload_decision.DE;
import utils.FormatData;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

@Slf4j
public class TestPartUnload {

    static List<Task> taskList = new ArrayList<>();

    static RoadsideUnit rsu = new RoadsideUnit();

    static Cloud cloud = new Cloud();

    // 车辆数目
    static List<Vehicle> vehicleList = new ArrayList<>();

    static List<Integer> unloadArr = new ArrayList<>();
    static List<Double> unloadRatioArr = new ArrayList<>();
    static List<Integer> freqAllocArrLocal = new ArrayList<>();
    static List<Integer> freqAllocArrRemote = new ArrayList<>();

    static int taskSizeFromDB;
    static int vehicleSizeFromDB;

    @Test
    public void testDSPRCOA_FINAL() {
        log.info("\n");
        log.error("======== test DSPRCOA FINAL =========");

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

        // NOTE: 根据任务类别，初始化资源分配
        List<Integer> unloadDecisionArrInit = new ArrayList<>(); // 个数应该为 geneLength
        List<Double> unloadRatioArrInit = new ArrayList<>();
        List<Integer> freqAllocArrInit = new ArrayList<>();
        List<Integer> freqAllocArrLocalInit = new ArrayList<>();
        List<Integer> freqAllocArrRemoteInit = new ArrayList<>();
        // 初始化资源分配
        for (int i = 0; i < taskSizeFromDB; i++) {
            // unloadDecisionArrInit.add(new Random().nextInt(vehicleSizeFromDB) - 1);
            unloadDecisionArrInit.add(taskList.get(i).getVehicleID());
            unloadRatioArrInit.add(0.5);

            // NOTE: 根据 task 的 cluster_class 参数初始化
            int currTaskClusterID = taskList.get(i).getClusterID();  // 任务聚类 -> class_
            Random random = new Random();
            double ratio = random.nextDouble();
            int tempFreqDefault = (int) (ratio * TaskPolicy.TASK_FREQ_INIT_CLASS_DEFAULT);
            int tempFreqClass = 0;
            if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_0) {
                tempFreqClass = (int) ((1.0 - ratio) * TaskPolicy.TASK_FREQ_INIT_CLASS_0);
                // freqAllocArrInit.add(Constants.TASK_FREQ_INIT_CLASS_0);
                freqAllocArrInit.add(tempFreqDefault + tempFreqClass);
                freqAllocArrLocalInit.add(tempFreqDefault + tempFreqClass);
                freqAllocArrRemoteInit.add(tempFreqDefault + tempFreqClass);
            } else if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_1) {
                tempFreqClass = (int) ((1.0 - ratio) * TaskPolicy.TASK_FREQ_INIT_CLASS_1);
                // freqAllocArrInit.add(Constants.TASK_FREQ_INIT_CLASS_1);
                freqAllocArrInit.add(tempFreqDefault + tempFreqClass);
                freqAllocArrLocalInit.add(tempFreqDefault + tempFreqClass);
                freqAllocArrRemoteInit.add(tempFreqDefault + tempFreqClass);
            } else if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_2) {
                tempFreqClass = (int) ((1.0 - ratio) * TaskPolicy.TASK_FREQ_INIT_CLASS_2);
                // freqAllocArrInit.add(Constants.TASK_FREQ_INIT_CLASS_2);
                freqAllocArrInit.add(tempFreqDefault + tempFreqClass);
                freqAllocArrLocalInit.add(tempFreqDefault + tempFreqClass);
                freqAllocArrRemoteInit.add(tempFreqDefault + tempFreqClass);
            } else if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_3) {
                tempFreqClass = (int) ((1.0 - ratio) * TaskPolicy.TASK_FREQ_INIT_CLASS_3);
                // freqAllocArrInit.add(Constants.TASK_FREQ_INIT_CLASS_3);
                freqAllocArrInit.add(tempFreqDefault + tempFreqClass);
                freqAllocArrLocalInit.add(tempFreqDefault + tempFreqClass);
                freqAllocArrRemoteInit.add(tempFreqDefault + tempFreqClass);
            } else {
                freqAllocArrInit.add(new Random().nextInt(TaskPolicy.TASK_FREQ_INIT_CLASS_DEFAULT) + 50);
                freqAllocArrLocalInit.add(tempFreqDefault + tempFreqClass);
                freqAllocArrRemoteInit.add(tempFreqDefault + tempFreqClass);
            }
        }

        // note：freqRemain ==> bounds[i][1] = freqRemain[unloadArr.get(i) + 1]
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
        int populationSize = 50;  // 个体数量
        int maxGenerations = 100;  // 最大迭代次数
        double crossoverRate = 0.8;  // 交叉概率
        double mutationFactor = 0.5;  // 变异概率

        // SA 参数
        double temperature = 100.0;  // 初始温度
        double coolingRate = 0.95;  // 冷却系数
        int stepsPerTemp = 100;  // 单温度迭代次数
        double minChange = 0.1;  // 判出条件

        // PSO 参数
        int numParticles = 50; // 粒子数量
        int numIterations = 100; // 迭代次数
        double inertiaWeight = 0.5; // 惯性权重
        double cognitiveWeight = 1.5; // 个体认知权重
        double socialWeight = 1.5; // 群体社会权重
        double[][] bounds = new double[numDimensions * 2][2];  // 边界
        for (int i = 0; i < numDimensions * 2; i++) {
            bounds[i][0] = 100;
            bounds[i][1] = 500;
        }


        // 记录目标list
        List<Double> uss_avg_list = new ArrayList<>();
        List<Double> energy_avg_list = new ArrayList<>();

        boolean flag_stop = false;
        while (!flag_stop) {
            for (int r = 0; r < maxRound; r++) {
                log.info("--- 第" + (r + 1) + "次迭代 ---");

                // DE
                DE de = new DE(populationSize, maxGenerations, crossoverRate,
                        mutationFactor, numDimensions, maxBound,
                        unloadDecisionArrInit, unloadRatioArrInit,
                        freqAllocArrLocalInit, freqAllocArrRemoteInit);
                de.optimize();
                int paretoSize = 0;
                paretoSize = de.getParetoFront().size();
                // log.info("DE - pareto 个数：" + paretoSize);
                // 随机选择一个（轮盘赌）
                // TODO：USSF 选择策略
                int sel_index_ussf = 0;
                double[] ussf_ga = new double[paretoSize];
                //
                double ussMax = getMaxValue(de.getParetoFront(), 0);
                double ussMin = getMinValue(de.getParetoFront(), 0);
                double ussDelta = ussMax - ussMin;
                double energyMax = getMaxValue(de.getParetoFront(), 1);
                double energyMin = getMinValue(de.getParetoFront(), 1);
                double energDelta = energyMax - energyMin;
                double xi = 0.7;
                double ussf_sum = 0;
                for (int i = 0; i < paretoSize; i++) {
                    ussf_ga[i] = xi * (de.getParetoFront().get(i)[0] - ussMin) / ussDelta
                            + (1 - xi) * (de.getParetoFront().get(i)[1] - energyMin) / energDelta;
                    ussf_sum += ussf_ga[i];
                }
                double rand = new Random().nextDouble();
                rand = rand * ussf_sum;
                double ussf_sel = 0;
                for (int i = 0; i < paretoSize; i++) {
                    ussf_sel += ussf_ga[i];
                    if (ussf_sel >= rand)  {
                        sel_index_ussf = i;
                        break;
                    }
                }
                int[] solutionSelect = de.getParetoFrontSolution().get(sel_index_ussf);
                // 卸载决策 --- 确定
                unloadArr.clear();
                for (int i = 0; i < numDimensions; i++) {
                    unloadArr.add(solutionSelect[i]);
                }

                // ---- **** 卸载比例
                SA sa = new SA(numDimensions, temperature, coolingRate, stepsPerTemp, minChange,
                        unloadArr, freqAllocArrLocalInit, freqAllocArrRemoteInit);
                sa.optimize();
                paretoSize = sa.getParetoFront().size();
                // log.info("SA - pareto 个数：" + paretoSize);
                ussf_ga = new double[paretoSize];
                ussMax = getMaxValue(sa.getParetoFront(), 0);
                ussMin = getMinValue(sa.getParetoFront(), 0);
                ussDelta = ussMax - ussMin;
                energyMax = getMaxValue(sa.getParetoFront(), 1);
                energyMin = getMinValue(sa.getParetoFront(), 1);
                energDelta = energyMax - energyMin;
                ussf_sum = 0;
                for (int i = 0; i < paretoSize; i++) {
                    ussf_ga[i] = xi * (sa.getParetoFront().get(i)[0] - ussMin) / ussDelta
                            + (1 - xi) * (sa.getParetoFront().get(i)[1] - energyMin) / energDelta;
                    ussf_sum += ussf_ga[i];
                }
                rand = new Random().nextDouble();
                rand = rand * ussf_sum;
                ussf_sel = 0;
                for (int i = 0; i < paretoSize; i++) {
                    ussf_sel += ussf_ga[i];
                    if (ussf_sel >= rand)  {
                        sel_index_ussf = i;
                        break;
                    }
                }
                sel_index_ussf = Math.min(sel_index_ussf, paretoSize - 1);
                double[] sol = sa.getParetoFrontRatioArr().get(sel_index_ussf);
                unloadRatioArrInit.clear();
                for (int i = 0; i < numDimensions; i++) {
                    unloadRatioArrInit.add(sol[i]);
                }

                // 重新定义粒子位置边界
                for (int i = 0; i < numDimensions; i++) {
                    bounds[i][0] = AlgorithmParam.MIN_POS_PARTICLE;
                    // bounds[i][1] = 500;
                    bounds[i][1] = Math.min(AlgorithmParam.MAX_POS_PARTICLE, freqRemain[unloadArr.get(i) + 1]);
                }

                PSO pso = new PSO(numParticles, numDimensions * 2, bounds,
                        inertiaWeight, cognitiveWeight, socialWeight,
                        unloadArr, unloadRatioArrInit);
                pso.optimize(numIterations);

                paretoSize = pso.getParetoFront().size();
                log.info("PSO - pareto 个数：" + paretoSize);
                ussf_ga = new double[paretoSize];
                ussMax = getMaxValue(pso.getParetoFront(), 0);
                ussMin = getMinValue(pso.getParetoFront(), 0);
                ussDelta = ussMax - ussMin;
                energyMax = getMaxValue(pso.getParetoFront(), 1);
                energyMin = getMinValue(pso.getParetoFront(), 1);
                energDelta = energyMax - energyMin;
                ussf_sum = 0;
                for (int i = 0; i < paretoSize; i++) {
                    ussf_ga[i] = xi * (pso.getParetoFront().get(i)[0] - ussMin) / ussDelta
                            + (1 - xi) * (pso.getParetoFront().get(i)[1] - energyMin) / energDelta;
                    ussf_sum += ussf_ga[i];
                }
                rand = new Random().nextDouble();
                rand = rand * ussf_sum;
                ussf_sel = 0;
                for (int i = 0; i < paretoSize; i++) {
                    ussf_sel += ussf_ga[i];
                    if (ussf_sel >= rand)  {
                        sel_index_ussf = i;
                        break;
                    }
                }

                double[] freqArr = pso.getParetoFrontPos().get(sel_index_ussf);
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

            for (double[] solution : paretoFrontList) {
                // log.info("USS avg val: " + solution[0] + ", Energy avg val: " + (-solution[1]));
                uss_avg_list.add(solution[0]);
                energy_avg_list.add(-solution[1]);
            }

            if (uss_avg_list.size() >= 200) {
                flag_stop = true;
            }
        }


        log.warn("★★★★★ 找到的 Pareto 最优解: ★★★★★");
        log.warn("uss_avg_list");
        log.info(uss_avg_list.toString());
        log.warn("energy_avg_list");
        log.info(energy_avg_list.toString());

        log.info("UnloadList: \n" + Arrays.toString(paretoUnloadList.get(0)));
        log.info("RatioList: \n" + Arrays.toString(FormatData.getEffectiveValue4Digit(paretoUnloadRatioList.get(0), 3)));

    }

    public static double getMaxValue(List<double[]> list, int index) {
        double res = list.get(0)[index];
        int size = list.size();
        for (int i = 0; i < size; i++) {
            res = Math.max(res, list.get(i)[index]);
        }
        return res;
    }

    public static double getMinValue(List<double[]> list, int index) {
        double res = list.get(0)[index];
        int size = list.size();
        for (int i = 0; i < size; i++) {
            res = Math.min(res, list.get(i)[index]);
        }
        return res;
    }

}
