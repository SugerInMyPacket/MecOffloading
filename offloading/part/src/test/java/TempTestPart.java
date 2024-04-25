import config.InitFrame;
import entity.Cloud;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import enums.AlgorithmParam;
import enums.Constants;
import enums.TaskPolicy;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import ratio_division.SA;
import ratio_division.SA_01;
import resource_allocation.PSO;
import unload_decision.DE;
import unload_decision.GA;
import utils.FormatData;
import utils.NumUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

@Slf4j(topic = "Temp - Test - Part")
public class TempTestPart {
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

    // 初始化任务
    public static void initTaskList() {
        for (int i = 0; i < len; i++) {
            Task newTask = new Task();
            newTask.setTaskID(i);
            newTask.setS(100);
            newTask.setR(0.1f);
            newTask.setC(10);
            newTask.setD(1000);
            newTask.setFactor(5);
            newTask.setI(1);
            newTask.setP(3);
            // newTask.setVehicleID(new Random().nextInt(vehicleNums));
            newTask.setVehicleID(2);
            taskList.add(newTask);
        }
    }

    // 初始化资源
    public static void initResources() {
        // 初始化 RSU 的信息
        rsu.setFreqMax(10000);
        rsu.setFreqRemain(500);

        // 初始化车辆
        for (int i = 0; i < vehicleNums; i++) {
            Vehicle vehicle = new Vehicle();
            vehicle.setVehicleID(i + 1);
            vehicle.setFreqMax(200);
            vehicle.setFreqRemain(200);

            vehicleList.add(vehicle);
        }
    }

    // 初始化 卸载决策和资源分配
    public static void initLamdaVar() {
        for (int i = 0; i < len; i++) {
            // unloadArr.add(-1);
            // unloadArr.add(1);
            // unloadArr.add(i % 4 + 1);
            // 把每个任务安排到每个车辆本地
            unloadArr.add(taskList.get(i).getVehicleID());
            // unloadArr.add(new Random().nextInt(vehicleNums + 1) - 1);
            // freqAllocArr.add(new Random().nextInt(100) + 50);
            unloadRatioArr.add(NumUtil.random(0.0, 1.0));
            freqAllocArrLocal.add(100);
            freqAllocArrRemote.add(100);
        }
    }

    /**
    * @Data 2024-03-18
    */
    @Test
    public void testAlgorithm() {
        // 初始化框架
        InitFrame.initFromDB();

        // 读取数据信息
        taskList = InitFrame.getTaskList();
        vehicleList = InitFrame.getVehicleList();
        rsu = InitFrame.getRSU();
        cloud = InitFrame.getCloud();

        taskSizeFromDB = taskList.size();
        log.warn("taskSizeFromDB: " + taskSizeFromDB);
        vehicleSizeFromDB = vehicleList.size();
        log.warn("vehicleSizeFromDB: " + vehicleSizeFromDB);

        // 初始化输入
        for (int i = 0; i < taskSizeFromDB; i++) {
            unloadArr.add(new Random().nextInt(vehicleSizeFromDB) - 1);
            unloadRatioArr.add(NumUtil.random(0.0, 1.0));
            freqAllocArrLocal.add(200);
            freqAllocArrRemote.add(200);
        }

        // 测试算法
        log.info("TempTestPart ===> 测试 DE algorithm .......");
        int populationSize_DE = 100;
        int maxGenerations = 100;
        double crossoverRate_DE = 0.8;
        double mutationFactor = 0.5;
        int dimensions = taskSizeFromDB;
        int maxBound = vehicleSizeFromDB;

        DE de = new DE(populationSize_DE, maxGenerations, crossoverRate_DE,
                mutationFactor, dimensions, maxBound,
                unloadRatioArr, freqAllocArrLocal, freqAllocArrRemote);
        de.optimize();
        log.info("DE 当前最优Solution: " + Arrays.toString(de.getParetoFrontSolution().get(0)));


        log.info("TempTestPart ===> 测试 GA algorithm .......");
        int populationSize = 20; // 种群大小
        double crossoverRate = 0.8; // 交叉概率
        double mutationRate = 0.01; // 变异概率
        int geneLength = taskSizeFromDB; // 染色体长度(维度数)
        int numGenerations = 50; // 迭代代数
        int bound = vehicleSizeFromDB;

        GA ga = new GA(populationSize, crossoverRate, mutationRate, geneLength, bound,
                unloadRatioArr, freqAllocArrLocal, freqAllocArrRemote);
        ga.optimizeChromosomes(numGenerations); // 优化
        log.info("GA 当前最优染色体: " + Arrays.toString(ga.getParetoFrontGene().get(0)));

        log.info("TempTestPart ===> 测试 SA algorithm .......");
        int numDimensions = taskSizeFromDB;
        double temperature = 100.0;
        double coolingRate = 0.95;
        int stepsPerTemp = 50;
        double minChange = 0.01;
        SA sa = new SA(numDimensions, temperature, coolingRate, stepsPerTemp, minChange,
                unloadArr, freqAllocArrLocal, freqAllocArrRemote);
        sa.optimize();
        double[] ratios
                = FormatData.getEffectiveValue4Digit(sa.getParetoFrontRatioArr().get(0), 2);
        log.info("SA 当前最优比例: " + Arrays.toString(ratios));


        log.info("TempTestPart ===> 测试PSO算法........");
        int numParticles = 30; // 粒子数量
        // int numDimensions = taskSizeFromDB; // 问题的维度，根据实际问题设置
        int numIterations = 100; // 迭代次数
        double inertiaWeight = 0.5; // 惯性权重
        double cognitiveWeight = 1.5; // 个体认知权重
        double socialWeight = 1.5; // 群体社会权重
        double[][] bounds = new double[numDimensions * 2][2];
        for (int i = 0; i < numDimensions * 2; i++) {
            bounds[i][0] = 100;
            bounds[i][1] = 500;
        }
        PSO pso = new PSO(numParticles, numDimensions * 2, bounds,
                inertiaWeight, cognitiveWeight, socialWeight,
                unloadArr, unloadRatioArr);
        double[] freq = FormatData.getEffectiveValue4Digit(pso.getParetoFrontPos().get(0), 0);
        log.info("PSO 最优粒子：" + Arrays.toString(freq));
    }


    /**
    * @Data 2024-03-20
    */
    @Test
    public void testDSPRCOA() {
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

        // NOTE: 根据任务类别，初始化资源分配
        List<Integer> unloadDecisionArrInit = new ArrayList<>(); // 个数应该为 geneLength
        List<Double> unloadRatioArrInit = new ArrayList<>();
        List<Integer> freqAllocArrInit = new ArrayList<>();
        List<Integer> freqAllocArrLocalInit = new ArrayList<>();
        List<Integer> freqAllocArrRemoteInit = new ArrayList<>();
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

            // freqAllocArr.add(freqAllocArrInit.get(i));
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
        int populationSize = 100;  // 个体数量
        int maxGenerations = 100;  // 最大迭代次数
        double crossoverRate = 0.8;  // 交叉概率
        double mutationFactor = 0.5;  // 变异概率

        // SA 参数
        double temperature = 100.0;  // 初始温度
        double coolingRate = 0.95;  // 冷却系数
        int stepsPerTemp = 100;  // 单温度迭代次数
        double minChange = 0.1;  // 判出条件

        // PSO 参数
        int numParticles = 30; // 粒子数量
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
            log.info("--- 第" + (r + 1) + "次迭代 ---");

            // DE
            DE de = new DE(populationSize, maxGenerations, crossoverRate,
                    mutationFactor, numDimensions, maxBound,
                    unloadDecisionArrInit, unloadRatioArrInit,
                    freqAllocArrLocalInit, freqAllocArrRemoteInit);
            de.optimize();
            int paretoSize = 0;
            paretoSize = de.getParetoFront().size();
            log.info("DE - pareto 个数：" + paretoSize);
            // 随机选择一个（轮盘赌）
            int selIndex = random.nextInt(paretoSize);
            int[] solutionSelect = de.getParetoFrontSolution().get(selIndex);

            unloadArr.clear();
            for (int i = 0; i < numDimensions; i++) {
                unloadArr.add(solutionSelect[i]);
            }

            SA sa = new SA(numDimensions, temperature, coolingRate, stepsPerTemp, minChange,
                    unloadArr, freqAllocArrLocalInit, freqAllocArrRemoteInit);
            sa.optimize();
            paretoSize = sa.getParetoFront().size();
            log.info("SA - pareto 个数：" + paretoSize);
            // TODO: 修改为找USS最高的那个解
            selIndex = random.nextInt(paretoSize);
            double tempUSS = sa.getParetoFront().get(0)[0];
            for (int j = 0; j < paretoSize; j++) {
                if (tempUSS < sa.getParetoFront().get(j)[0]) {
                    tempUSS = sa.getParetoFront().get(j)[0];
                    selIndex = j;
                }
            }
            double[] sol = sa.getParetoFrontRatioArr().get(selIndex);
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
            selIndex = random.nextInt(paretoSize);
            tempUSS = pso.getParetoFront().get(0)[0];
            for (int j = 0; j < paretoSize; j++) {
                if (tempUSS < pso.getParetoFront().get(j)[0]) {
                    tempUSS = pso.getParetoFront().get(j)[0];
                    selIndex = j;
                }
            }
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

        log.info("UnloadList: \n" + Arrays.toString(paretoUnloadList.get(0)));
        log.info("RatioList: \n" + Arrays.toString(FormatData.getEffectiveValue4Digit(paretoUnloadRatioList.get(0), 3)));

    }

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

            // freqAllocArr.add(freqAllocArrInit.get(i));
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
            log.info("DE - pareto 个数：" + paretoSize);
            // 随机选择一个（轮盘赌）
            // TODO：USSF 选择策略
            int sel_index_ussf = 0;
            int gaParetoSize = de.getParetoFront().size();
            double[] ussf_ga = new double[gaParetoSize];
            //
            double ussMax = getMaxValue(de.getParetoFront(), 0);
            double ussMin = getMinValue(de.getParetoFront(), 0);
            double ussDelta = ussMax - ussMin;
            double energyMax = getMaxValue(de.getParetoFront(), 1);
            double energyMin = getMinValue(de.getParetoFront(), 1);
            double energDelta = energyMax - energyMin;
            double xi = 0.7;
            double ussf_sum = 0;
            for (int i = 0; i < gaParetoSize; i++) {
                ussf_ga[i] = xi * (de.getParetoFront().get(i)[0] - ussMin) / ussDelta
                        + (1 - xi) * (de.getParetoFront().get(i)[1] - energyMin) / energDelta;
                ussf_sum += ussf_ga[i];
            }
            double rand = new Random().nextDouble();
            rand = rand * ussf_sum;
            double ussf_sel = 0;
            for (int i = 0; i < gaParetoSize; i++) {
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
            log.info("SA - pareto 个数：" + paretoSize);

            ussMax = getMaxValue(sa.getParetoFront(), 0);
            ussMin = getMinValue(sa.getParetoFront(), 0);
            ussDelta = ussMax - ussMin;
            energyMax = getMaxValue(sa.getParetoFront(), 1);
            energyMin = getMinValue(sa.getParetoFront(), 1);
            energDelta = energyMax - energyMin;
            ussf_sum = 0;
            for (int i = 0; i < gaParetoSize; i++) {
                ussf_ga[i] = xi * (sa.getParetoFront().get(i)[0] - ussMin) / ussDelta
                        + (1 - xi) * (sa.getParetoFront().get(i)[1] - energyMin) / energDelta;
                ussf_sum += ussf_ga[i];
            }
            rand = new Random().nextDouble();
            rand = rand * ussf_sum;
            ussf_sel = 0;
            for (int i = 0; i < gaParetoSize; i++) {
                ussf_sel += ussf_ga[i];
                if (ussf_sel >= rand)  {
                    sel_index_ussf = i;
                    break;
                }
            }

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

            ussMax = getMaxValue(pso.getParetoFront(), 0);
            ussMin = getMinValue(pso.getParetoFront(), 0);
            ussDelta = ussMax - ussMin;
            energyMax = getMaxValue(pso.getParetoFront(), 1);
            energyMin = getMinValue(pso.getParetoFront(), 1);
            energDelta = energyMax - energyMin;
            ussf_sum = 0;
            for (int i = 0; i < gaParetoSize; i++) {
                ussf_ga[i] = xi * (pso.getParetoFront().get(i)[0] - ussMin) / ussDelta
                        + (1 - xi) * (pso.getParetoFront().get(i)[1] - energyMin) / energDelta;
                ussf_sum += ussf_ga[i];
            }
            rand = new Random().nextDouble();
            rand = rand * ussf_sum;
            ussf_sel = 0;
            for (int i = 0; i < gaParetoSize; i++) {
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

        log.warn("★★★★★ 找到的 Pareto 最优解: ★★★★★");
        for (double[] solution : paretoFrontList) {
            log.info("USS avg val: " + solution[0] + ", Energy avg val: " + (-solution[1]));
        }

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
