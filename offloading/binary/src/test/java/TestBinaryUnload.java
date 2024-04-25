import config.InitFrame;
import entity.Cloud;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import enums.Constants;
import enums.TaskPolicy;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import resource_allocation.PSO_05;
import resource_allocation.PSO_06;
import unload_decision.GA_04;
import utils.FormatData;
import utils.Formula;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleToIntFunction;

@Slf4j(topic = "BinaryUnload_Test")
public class TestBinaryUnload {

    static List<Task> taskList = new ArrayList<>();

    static RoadsideUnit rsu = new RoadsideUnit();
    static Cloud cloud = new Cloud();
    static List<Vehicle> vehicleList = new ArrayList<>();

    // 任务数量
    static int taskSizeFromDB;
    // 车辆数目
    static int vehicleSizeFromDB;

    // 卸载决策
    static List<Integer> unloadArr = new ArrayList<>();
    // 资源分配
    static List<Integer> freqAllocArr = new ArrayList<>();


    /**
    * @Data 2024-03-06
    */
    @Test
    public void testBinaryRHGPCA() {
        log.error("###### ==== testBinary - RHGPCA === ######");

        // 初始化
        InitFrame.initFromDB();

        // 读取信息
        taskList = InitFrame.getTaskList();
        vehicleList = InitFrame.getVehicleList();
        rsu = InitFrame.getRSU();
        cloud = InitFrame.getCloud();

        taskSizeFromDB = taskList.size();
        vehicleSizeFromDB = vehicleList.size();
        System.out.println("taskSizeFromDB: " + taskSizeFromDB);
        System.out.println("vehicleSizeFromDB: " + vehicleSizeFromDB);

        // TODO: 根据任务类别，初始化资源分配

        int maxRound = 10;

        log.info(" ===> 测试 GA algorithm .......");
        int populationSize = 20; // 种群大小
        double crossoverRate = 0.8; // 交叉概率
        double mutationRate = 0.01; // 变异概率
        int geneLength = taskSizeFromDB; // 染色体长度(维度数)
        int numGenerations = Constants.NUM_ITERATIONS; // 迭代代数
        int bound = vehicleSizeFromDB;

        List<Integer> unloadDecisionArrInit = new ArrayList<>(); // 个数应该为 geneLength
        List<Integer> freqAllocArrInit = new ArrayList<>();
        for (int i = 0; i < geneLength; i++) {
            unloadDecisionArrInit.add(new Random().nextInt(vehicleSizeFromDB) - 1);
            freqAllocArrInit.add(new Random().nextInt(200) + 50);
        }

        GA_04 ga = new GA_04(populationSize, crossoverRate, mutationRate, geneLength, bound - 1, freqAllocArrInit);
        ga.optimizeChromosomes(numGenerations); // 优化

        System.out.println("=================GA 输出=================");
        System.out.println(Arrays.toString(ga.getParetoFront().get(0)));
        for (int[] arr : ga.getParetoFrontGene()) {
            System.out.println(Arrays.toString(arr));
            for (int i = 0; i < arr.length; i++) {
                unloadArr.add(arr[i]);
            }
            break;
        }

        log.info(" ===> 测试PSO算法........");
        int numParticles = 20; // 粒子数量
        int numDimensions = taskSizeFromDB; // 问题的维度，根据实际问题设置
        int numIterations = Constants.NUM_ITERATIONS; // 迭代次数
        double inertiaWeight = 0.5; // 惯性权重
        double cognitiveWeight = 1.5; // 个体认知权重
        double socialWeight = 1.5; // 群体社会权重
        // double[][] bounds = {{10, 500}, {10, 500}};
        double[][] bounds = new double[numDimensions][2];

        // note：freqRemain ==> bounds[i][1] = freqRemain[unloadArr.get(i) + 1]
        int[] freqRemain = new int[vehicleSizeFromDB + 2];
        freqRemain[0] = (int) cloud.getFreqRemain();
        freqRemain[1] = (int) rsu.getFreqRemain();
        for (int i = 0; i < vehicleSizeFromDB; i++) {
            freqRemain[i + 2] = (int) vehicleList.get(i).getFreqRemain();
        }

        System.out.println("*****************************************");
        System.out.println(Arrays.toString(freqRemain));
        System.out.println("*****************************************");

        for (int i = 0; i < numDimensions; i++) {
            bounds[i][0] = 100;
            // bounds[i][1] = 500;
            bounds[i][1] = Math.min(1000, freqRemain[unloadArr.get(i) + 1]);
        }

        PSO_06 p = new PSO_06(numParticles, numDimensions, bounds, inertiaWeight, cognitiveWeight,
                socialWeight, unloadArr);
        p.optimize(numIterations);

        // System.out.println(unloadArr);
        for (double[] res : p.getParetoFrontPos()) {
            int[] array = Arrays.stream(res).mapToInt(new DoubleToIntFunction() {
                @Override
                public int applyAsInt(double value) {
                    return (int) value;
                }
            }).toArray();
            // System.out.println("resAlloc:" + Arrays.toString(array));
            for (int i = 0; i < array.length; i++) {
                freqAllocArr.add(array[i]);
            }
            break;
        }

        log.warn("★★★★★ GA-找到的Pareto最优解: ★★★★★");
        for (double[] solution : ga.getParetoFront()) {
            log.warn("USS val: " + solution[0] + ", Energy val: " + -solution[1]);
        }
        log.info("ParetoFrontGene().size() : " + ga.getParetoFrontGene().size());

        log.warn("★★★★★ PSO后-找到的Pareto最优解: ★★★★★");
        for (double[] solution : p.getParetoFront()) {
            log.info("USS val: " + solution[0] + ", Energy val: " + -solution[1]);
        }

        log.info("ParetoFrontPos().size() : " + p.getParetoFrontPos().size());
        List<int[]> paretoUnloadArr = p.getParetoUnloadArr();
        log.info("final unload arr : " + paretoUnloadArr.size() + "\n" + Arrays.toString(paretoUnloadArr.get(paretoUnloadArr.size() - 1)));
        log.info("final unload arr2 : \n" + Arrays.toString(unloadArr.toArray()));
        log.info("final resAlloc arr : \n" + Arrays.toString(freqAllocArr.toArray()));

        int[] paretoUnloadArrLast = p.getParetoUnloadArr().get(p.getParetoUnloadArr().size() - 1);
        for (int i = 0; i < taskSizeFromDB; i++) {
            // freqRemain[unloadArr.get(i) + 1] = freqRemain[unloadArr.get(i) + 1] - freqAllocArr.get(i);
            freqRemain[paretoUnloadArrLast[i] + 1] = freqRemain[paretoUnloadArrLast[i] + 1] - freqAllocArr.get(i);
        }
        log.info("freq remain arr : \n" + Arrays.toString(freqRemain));

        log.info("******************** task cost time *******************");
        List<Integer> tempUnloadArr = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            tempUnloadArr.add(paretoUnloadArrLast[i]);
        }
        List<Double> taskCostTimeList = new ArrayList<>();
        List<Double> taskUplink2RSUTimeList = new ArrayList<>();
        List<Double> taskUssList = new ArrayList<>();
        Formula.calculateTaskCostTime(taskList, tempUnloadArr, freqAllocArr, taskCostTimeList);
        Formula.calculateTaskTransTime4Uplink2RSU(taskList, tempUnloadArr, taskUplink2RSUTimeList);
        taskUssList = Formula.getUSS4TaskList(taskList, tempUnloadArr, freqAllocArr);
        for (int i = 0; i < taskSizeFromDB; i++) {
            taskUplink2RSUTimeList.set(i, FormatData.getEffectiveValue4Digit(taskUplink2RSUTimeList.get(i) * 100.0, 2));
            taskCostTimeList.set(i, FormatData.getEffectiveValue4Digit(taskCostTimeList.get(i) * 100.0, 2));
            taskUssList.set(i, FormatData.getEffectiveValue4Digit(taskUssList.get(i), 2));
        }

        log.info("taskUplink2RSUTimeList: \n" + taskUplink2RSUTimeList);
        log.info("taskCostTimeList: \n" + taskCostTimeList);
        log.info("taskUssList: \n" + taskUssList);
    }

    @Test
    public void testBinaryRHGPCA2() {

        log.info("\n");
        log.error("###### ==== testBinary - RHGPCA - 2 === ######");

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
        List<Integer> freqAllocArrInit = new ArrayList<>();
        for (int i = 0; i < taskSizeFromDB; i++) {
            // unloadDecisionArrInit.add(new Random().nextInt(vehicleSizeFromDB) - 1);
            unloadDecisionArrInit.add(taskList.get(i).getVehicleID());

            // NOTE: 根据 task 的 cluster_class 参数初始化
            int currTaskClusterID = taskList.get(i).getClusterID();  // 任务聚类 -> class_
            Random random = new Random();
            double ratio = random.nextDouble();
            int tempFreqDefault = (int) (ratio * TaskPolicy.TASK_FREQ_INIT_CLASS_DEFAULT);
            int tempFreqClass = 0;
            if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_0) {
                // tempFreqClass = (int) ((1.0 - ratio) * Constants.TASK_FREQ_INIT_CLASS_0);
                freqAllocArrInit.add(TaskPolicy.TASK_FREQ_INIT_CLASS_0);
                freqAllocArrInit.add(tempFreqDefault + tempFreqClass);
            } else if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_1) {
                // tempFreqClass = (int) ((1.0 - ratio) * Constants.TASK_FREQ_INIT_CLASS_1);
                freqAllocArrInit.add(TaskPolicy.TASK_FREQ_INIT_CLASS_1);
                freqAllocArrInit.add(tempFreqDefault + tempFreqClass);
            } else if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_2) {
                // tempFreqClass = (int) ((1.0 - ratio) * Constants.TASK_FREQ_INIT_CLASS_2);
                freqAllocArrInit.add(TaskPolicy.TASK_FREQ_INIT_CLASS_2);
                freqAllocArrInit.add(tempFreqDefault + tempFreqClass);
            } else if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_3) {
                freqAllocArrInit.add(TaskPolicy.TASK_FREQ_INIT_CLASS_3);
                // tempFreqClass = (int) ((1.0 - ratio) * Constants.TASK_FREQ_INIT_CLASS_3);
                freqAllocArrInit.add(tempFreqDefault + tempFreqClass);
            } else {
                freqAllocArrInit.add(new Random().nextInt(TaskPolicy.TASK_FREQ_INIT_CLASS_DEFAULT) + 50);
            }

            freqAllocArr.add(freqAllocArrInit.get(i));
        }

        // 大循环迭代次数
        int maxRound = Constants.NUM_ROUND_TIMES;

        // GA 参数
        int populationSize = Constants.NUM_CHROMOSOMES; // 种群大小
        double crossoverRate = 0.8; // 交叉概率
        double mutationRate = 0.01; // 变异概率
        int geneLength = taskSizeFromDB; // 染色体长度(维度数)
        int numGenerations = Constants.NUM_ITERATIONS; // 迭代代数
        int bound = vehicleSizeFromDB;

        // PSO 参数
        int numParticles = Constants.NUM_PARTICLES; // 粒子数量
        int numDimensions = taskSizeFromDB; // 问题的维度，根据实际问题设置
        int numIterations = Constants.NUM_ITERATIONS; // 迭代次数
        double inertiaWeight = 0.5; // 惯性权重
        double cognitiveWeight = 1.5; // 个体认知权重
        double socialWeight = 1.5; // 群体社会权重
        // double[][] bounds = {{100, 500}, {100, 500}};
        double[][] bounds = new double[numDimensions][2];

        // note：freqRemain ==> bounds[i][1] = freqRemain[unloadArr.get(i) + 1]
        int[] freqRemain = new int[vehicleSizeFromDB + 2];
        freqRemain[0] = (int) cloud.getFreqRemain();
        freqRemain[1] = (int) rsu.getFreqRemain();
        for (int i = 0; i < vehicleSizeFromDB; i++) {
            freqRemain[i + 2] = (int) vehicleList.get(i).getFreqRemain();
        }
        log.info("Init Freq Remain: \n" + Arrays.toString(freqRemain));

        // round start ...
        List<int[]> paretoUnloadList = new ArrayList<>();
        List<double[]> paretoFrontList = new ArrayList<>();
        List<double[]> paretoFrontPosList = new ArrayList<>();

        for (int round = 0; round < maxRound; round++) {
            // 卸载决策
            GA_04 ga = new GA_04(populationSize, crossoverRate, mutationRate, geneLength,
                    bound - 1, unloadDecisionArrInit, freqAllocArrInit);
            ga.optimizeChromosomes(numGenerations); // 优化

            // TODO：USSF 选择策略
            int sel_index = 0;
            int gaParetoSize = ga.getParetoFront().size();
            double[] ussf_ga = new double[gaParetoSize];
            //
            double ussMax = getMaxValue(ga.getParetoFront(), 0);
            double ussMin = getMinValue(ga.getParetoFront(), 0);
            double ussDelta = ussMax - ussMin;
            double energyMax = getMaxValue(ga.getParetoFront(), 1);
            double energyMin = getMinValue(ga.getParetoFront(), 1);
            double energDelta = energyMax - energyMin;
            double xi = 0.6;
            double ussf_sum = 0;
            for (int i = 0; i < gaParetoSize; i++) {
                ussf_ga[i] = xi * (ga.getParetoFront().get(i)[0] - ussMin) / ussDelta
                        + (1 - xi) * (ga.getParetoFront().get(i)[1] - energyMin) / energDelta;
                ussf_sum += ussf_ga[i];
            }
            double rand = new Random().nextDouble();
            rand = rand * ussf_sum;
            double ussf_sel = 0;
            for (int i = 0; i < gaParetoSize; i++) {
                ussf_sel += ussf_ga[i];
                if (ussf_sel >= rand)  {
                    sel_index = i;
                    break;
                }
            }
            int[] geneArr = ga.getParetoFrontGene().get(sel_index);
            unloadArr.clear();
            for (int i = 0; i < geneArr.length; i++) {
                unloadArr.add(geneArr[i]);
            }

            for (int i = 0; i < numDimensions; i++) {
                bounds[i][0] = Constants.MIN_POS_PARTICLE;
                // bounds[i][1] = 500;
                bounds[i][1] = Math.min(Constants.MAX_POS_PARTICLE, freqRemain[unloadArr.get(i) + 1]);
            }

            // 资源分配
            // PSO_06 p = new PSO_06(numParticles, numDimensions, bounds,
            //         inertiaWeight, cognitiveWeight,
            //         socialWeight, unloadArr);
            PSO_06 p = new PSO_06(numParticles, numDimensions, bounds,
                    inertiaWeight, cognitiveWeight, socialWeight,
                    unloadArr, freqAllocArr);
            p.optimize(numIterations);

            ussMax = getMaxValue(p.getParetoFront(), 0);
            ussMin = getMinValue(p.getParetoFront(), 0);
            ussDelta = ussMax - ussMin;
            energyMax = getMaxValue(p.getParetoFront(), 1);
            energyMin = getMinValue(p.getParetoFront(), 1);
            energDelta = energyMax - energyMin;
            ussf_sum = 0;
            for (int i = 0; i < gaParetoSize; i++) {
                ussf_ga[i] = xi * (p.getParetoFront().get(i)[0] - ussMin) / ussDelta
                        + (1 - xi) * (p.getParetoFront().get(i)[1] - energyMin) / energDelta;
                ussf_sum += ussf_ga[i];
            }
            rand = new Random().nextDouble();
            rand = rand * ussf_sum;
            ussf_sel = 0;
            for (int i = 0; i < gaParetoSize; i++) {
                ussf_sel += ussf_ga[i];
                if (ussf_sel >= rand)  {
                    sel_index = i;
                    break;
                }
            }
            // TODO: 可考虑随机取 or 权重取
            int[] unloadGetOne = p.getParetoUnloadArr().get(sel_index);
            double[] freqGetOne = p.getParetoFrontPos().get(sel_index);

            // 修改
            // unloadDecisionArrInit.clear();
            // freqAllocArrInit.clear();
            freqAllocArr.clear();
            for (int i = 0; i < numDimensions; i++) {
                // unloadDecisionArrInit.add(unloadGetOne[i]);
                // freqAllocArrInit.add((int) freqGetOne[i]);
                freqAllocArr.add((int) freqGetOne[i]);
            }

            // 记录
            if (round == maxRound - 1) {
                paretoUnloadList = p.getParetoUnloadArr();
                paretoFrontList = p.getParetoFront();
                paretoFrontPosList = p.getParetoFrontPos();
            }
            // round over !!!
        }

        log.warn("★★★★★ 找到的 Pareto 最优解: ★★★★★");
        int pareto_size = paretoFrontList.size();
        List<Double> uss_avg_list = new ArrayList<>();
        List<Double> energy_avg_list = new ArrayList<>();
        for (double[] solution : paretoFrontList) {
            uss_avg_list.add(solution[0]);
            energy_avg_list.add(-solution[1]);
        }
        log.warn("USS avg val: ");
        log.info(uss_avg_list.toString());
        log.warn("Energy avg val: ");
        log.info(energy_avg_list.toString());


        log.warn("ParetoFrontPos().size() : " + paretoFrontList.size());
        log.info("final unload arr : " + paretoUnloadList.size() + "\n"
                + Arrays.toString(paretoUnloadList.get(paretoUnloadList.size() - 1)));

        // for (int i = 0; i < numDimensions; i++) {
        //     freqAllocArr.add((int) paretoFrontPosList.get(paretoFrontPosList.size() - 1)[i]);
        // }
        log.info("final unload arr2 : \n" + Arrays.toString(unloadArr.toArray()));
        log.info("final resAlloc arr : \n" + Arrays.toString(freqAllocArr.toArray()));

        int[] paretoUnloadArrLast = paretoUnloadList.get(paretoUnloadList.size() - 1);
        for (int i = 0; i < taskSizeFromDB; i++) {
            // freqRemain[unloadArr.get(i) + 1] = freqRemain[unloadArr.get(i) + 1] - freqAllocArr.get(i);
            freqRemain[paretoUnloadArrLast[i] + 1]
                    = freqRemain[paretoUnloadArrLast[i] + 1] - freqAllocArr.get(i);
        }
        log.info("freq remain arr : \n" + Arrays.toString(freqRemain));

        log.info("******************** task cost time *******************");
        List<Integer> tempUnloadArr = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            tempUnloadArr.add(paretoUnloadArrLast[i]);
        }

        List<Double> taskCostTimeList = new ArrayList<>();
        List<Double> taskUplink2RSUTimeList = new ArrayList<>();
        List<Double> taskUplink2CloudTimeList = new ArrayList<>();
        List<Double> taskUssList = new ArrayList<>();
        Formula.calculateTaskCostTime(taskList, tempUnloadArr, freqAllocArr, taskCostTimeList);
        Formula.calculateTaskTransTime4Uplink2RSU(taskList, tempUnloadArr, taskUplink2RSUTimeList);
        Formula.calculateTaskTransTime4UplinkR2C(taskList, unloadArr, taskUplink2CloudTimeList);
        taskUssList = Formula.getUSS4TaskList(taskList, tempUnloadArr, freqAllocArr);
        for (int i = 0; i < taskSizeFromDB; i++) {
            taskUplink2RSUTimeList.set(i,
                    FormatData.getEffectiveValue4Digit(taskUplink2RSUTimeList.get(i) * 100.0, 2));
            taskUplink2CloudTimeList.set(i,
                    FormatData.getEffectiveValue4Digit(taskUplink2CloudTimeList.get(i) * 100.0, 2));
            taskCostTimeList.set(i,
                    FormatData.getEffectiveValue4Digit(taskCostTimeList.get(i) * 100.0, 2));
            taskUssList.set(i,
                    FormatData.getEffectiveValue4Digit(taskUssList.get(i), 2));
        }

        log.info("taskUplink2RSUTimeList: \n" + taskUplink2RSUTimeList);
        log.info("taskCostTimeList: \n" + taskCostTimeList);
        log.info("taskUssList: \n" + taskUssList);
    }


    @Test
    public void testBinaryRHGPCA_final() {

        log.info("\n");
        log.error("###### ==== testBinary - RHGPCA - Final === ######");

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
        List<Integer> freqAllocArrInit = new ArrayList<>();
        for (int i = 0; i < taskSizeFromDB; i++) {
            // unloadDecisionArrInit.add(new Random().nextInt(vehicleSizeFromDB) - 1);
            unloadDecisionArrInit.add(taskList.get(i).getVehicleID());

            // NOTE: 根据 task 的 cluster_class 参数初始化
            int currTaskClusterID = taskList.get(i).getClusterID();  // 任务聚类 -> class_
            Random random = new Random();
            double ratio = random.nextDouble();
            int tempFreqDefault = (int) (ratio * TaskPolicy.TASK_FREQ_INIT_CLASS_DEFAULT);
            int tempFreqClass = 0;
            if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_0) {
                // tempFreqClass = (int) ((1.0 - ratio) * Constants.TASK_FREQ_INIT_CLASS_0);
                freqAllocArrInit.add(TaskPolicy.TASK_FREQ_INIT_CLASS_0);
                freqAllocArrInit.add(tempFreqDefault + tempFreqClass);
            } else if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_1) {
                // tempFreqClass = (int) ((1.0 - ratio) * Constants.TASK_FREQ_INIT_CLASS_1);
                freqAllocArrInit.add(TaskPolicy.TASK_FREQ_INIT_CLASS_1);
                freqAllocArrInit.add(tempFreqDefault + tempFreqClass);
            } else if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_2) {
                // tempFreqClass = (int) ((1.0 - ratio) * Constants.TASK_FREQ_INIT_CLASS_2);
                freqAllocArrInit.add(TaskPolicy.TASK_FREQ_INIT_CLASS_2);
                freqAllocArrInit.add(tempFreqDefault + tempFreqClass);
            } else if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_3) {
                freqAllocArrInit.add(TaskPolicy.TASK_FREQ_INIT_CLASS_3);
                // tempFreqClass = (int) ((1.0 - ratio) * Constants.TASK_FREQ_INIT_CLASS_3);
                freqAllocArrInit.add(tempFreqDefault + tempFreqClass);
            } else {
                freqAllocArrInit.add(new Random().nextInt(TaskPolicy.TASK_FREQ_INIT_CLASS_DEFAULT) + 50);
            }

            freqAllocArr.add(freqAllocArrInit.get(i));
        }

        // 大循环迭代次数
        int maxRound = Constants.NUM_ROUND_TIMES;

        // GA 参数
        int populationSize = Constants.NUM_CHROMOSOMES; // 种群大小
        int numGenerations = Constants.NUM_ITERATIONS; // 迭代代数
        double crossoverRate = 0.8; // 交叉概率
        double mutationRate = 0.01; // 变异概率
        int geneLength = taskSizeFromDB; // 染色体长度(维度数)
        int bound = vehicleSizeFromDB;

        // PSO 参数
        int numParticles = Constants.NUM_PARTICLES; // 粒子数量
        int numDimensions = taskSizeFromDB; // 问题的维度，根据实际问题设置
        int numIterations = Constants.NUM_ITERATIONS; // 迭代次数
        double inertiaWeight = 0.5; // 惯性权重
        double cognitiveWeight = 1.5; // 个体认知权重
        double socialWeight = 1.5; // 群体社会权重
        // double[][] bounds = {{100, 500}, {100, 500}};
        double[][] bounds = new double[numDimensions][2];

        // note：freqRemain ==> bounds[i][1] = freqRemain[unloadArr.get(i) + 1]
        int[] freqRemain = new int[vehicleSizeFromDB + 2];
        freqRemain[0] = (int) cloud.getFreqRemain();
        freqRemain[1] = (int) rsu.getFreqRemain();
        for (int i = 0; i < vehicleSizeFromDB; i++) {
            freqRemain[i + 2] = (int) vehicleList.get(i).getFreqRemain();
        }
        log.info("Init Freq Remain: \n" + Arrays.toString(freqRemain));

        // round start ...
        List<int[]> paretoUnloadList = new ArrayList<>();
        List<double[]> paretoFrontList = new ArrayList<>();
        List<double[]> paretoFrontPosList = new ArrayList<>();

        for (int round = 0; round < maxRound; round++) {
            // 卸载决策
            GA_04 ga = new GA_04(populationSize, crossoverRate, mutationRate, geneLength,
                    bound - 1, unloadDecisionArrInit, freqAllocArrInit);
            ga.optimizeChromosomes(numGenerations); // 优化

            // TODO：USSF 选择策略
            int sel_index = 0;
            int gaParetoSize = ga.getParetoFront().size();
            double[] ussf_ga = new double[gaParetoSize];
            //
            double ussMax = getMaxValue(ga.getParetoFront(), 0);
            double ussMin = getMinValue(ga.getParetoFront(), 0);
            double ussDelta = ussMax - ussMin;
            double energyMax = getMaxValue(ga.getParetoFront(), 1);
            double energyMin = getMinValue(ga.getParetoFront(), 1);
            double energDelta = energyMax - energyMin;
            double xi = 0.6;
            double ussf_sum = 0;
            for (int i = 0; i < gaParetoSize; i++) {
                ussf_ga[i] = xi * (ga.getParetoFront().get(i)[0] - ussMin) / ussDelta
                        + (1 - xi) * (ga.getParetoFront().get(i)[1] - energyMin) / energDelta;
                ussf_sum += ussf_ga[i];
            }
            double rand = new Random().nextDouble();
            rand = rand * ussf_sum;
            double ussf_sel = 0;
            for (int i = 0; i < gaParetoSize; i++) {
                ussf_sel += ussf_ga[i];
                if (ussf_sel >= rand)  {
                    sel_index = i;
                    break;
                }
            }
            int[] geneArr = ga.getParetoFrontGene().get(sel_index);
            unloadArr.clear();
            for (int i = 0; i < geneArr.length; i++) {
                unloadArr.add(geneArr[i]);
            }

            for (int i = 0; i < numDimensions; i++) {
                bounds[i][0] = Constants.MIN_POS_PARTICLE;
                // bounds[i][1] = 500;
                bounds[i][1] = Math.min(Constants.MAX_POS_PARTICLE, freqRemain[unloadArr.get(i) + 1]);
            }

            // 资源分配
            // PSO_06 p = new PSO_06(numParticles, numDimensions, bounds,
            //         inertiaWeight, cognitiveWeight,
            //         socialWeight, unloadArr);
            PSO_06 p = new PSO_06(numParticles, numDimensions, bounds,
                    inertiaWeight, cognitiveWeight, socialWeight,
                    unloadArr, freqAllocArr);
            p.optimize(numIterations);

            ussMax = getMaxValue(p.getParetoFront(), 0);
            ussMin = getMinValue(p.getParetoFront(), 0);
            ussDelta = ussMax - ussMin;
            energyMax = getMaxValue(p.getParetoFront(), 1);
            energyMin = getMinValue(p.getParetoFront(), 1);
            energDelta = energyMax - energyMin;
            ussf_sum = 0;
            for (int i = 0; i < gaParetoSize; i++) {
                ussf_ga[i] = xi * (p.getParetoFront().get(i)[0] - ussMin) / ussDelta
                        + (1 - xi) * (p.getParetoFront().get(i)[1] - energyMin) / energDelta;
                ussf_sum += ussf_ga[i];
            }
            rand = new Random().nextDouble();
            rand = rand * ussf_sum;
            ussf_sel = 0;
            for (int i = 0; i < gaParetoSize; i++) {
                ussf_sel += ussf_ga[i];
                if (ussf_sel >= rand)  {
                    sel_index = i;
                    break;
                }
            }
            // TODO: 可考虑随机取 or 权重取
            int[] unloadGetOne = p.getParetoUnloadArr().get(sel_index);
            double[] freqGetOne = p.getParetoFrontPos().get(sel_index);

            // 修改
            // unloadDecisionArrInit.clear();
            // freqAllocArrInit.clear();
            freqAllocArr.clear();
            for (int i = 0; i < numDimensions; i++) {
                // unloadDecisionArrInit.add(unloadGetOne[i]);
                // freqAllocArrInit.add((int) freqGetOne[i]);
                freqAllocArr.add((int) freqGetOne[i]);
            }

            // 记录
            if (round == maxRound - 1) {
                paretoUnloadList = p.getParetoUnloadArr();
                paretoFrontList = p.getParetoFront();
                paretoFrontPosList = p.getParetoFrontPos();
            }
            // round over !!!
        }

        log.warn("★★★★★ 找到的 Pareto 最优解: ★★★★★");
        int pareto_size = paretoFrontList.size();
        log.warn("ParetoFrontPos().size() : " + pareto_size);
        List<Double> uss_avg_list = new ArrayList<>();
        List<Double> energy_avg_list = new ArrayList<>();
        for (double[] solution : paretoFrontList) {
            uss_avg_list.add(solution[0]);
            energy_avg_list.add(-solution[1]);
        }

        log.warn("USS avg val list: ");
        log.info(uss_avg_list.toString());
        log.warn("Energy avg val list: ");
        log.info(energy_avg_list.toString());

        log.warn("******************** task cost time *******************");
        List<Double> taskCostTimeList = new ArrayList<>();
        List<Double> taskUplink2RSUTimeList = new ArrayList<>();
        List<Double> taskUplink2CloudTimeList = new ArrayList<>();
        List<Double> taskUssList = new ArrayList<>();
        Formula.calculateTaskCostTime(taskList, unloadArr, freqAllocArr, taskCostTimeList);
        Formula.calculateTaskTransTime4Uplink2RSU(taskList, unloadArr, taskUplink2RSUTimeList);
        Formula.calculateTaskTransTime4UplinkR2C(taskList, unloadArr, taskUplink2CloudTimeList);
        taskUssList = Formula.getUSS4TaskList(taskList, unloadArr, freqAllocArr);
        for (int i = 0; i < taskSizeFromDB; i++) {
            taskUplink2RSUTimeList.set(i,
                    FormatData.getEffectiveValue4Digit(taskUplink2RSUTimeList.get(i) * 100.0, 2));
            taskUplink2CloudTimeList.set(i,
                    FormatData.getEffectiveValue4Digit(taskUplink2CloudTimeList.get(i) * 100.0, 2));
            taskCostTimeList.set(i,
                    FormatData.getEffectiveValue4Digit(taskCostTimeList.get(i) * 100.0, 2));
            taskUssList.set(i,
                    FormatData.getEffectiveValue4Digit(taskUssList.get(i), 2));
        }

        log.info("taskUplink2RSUTimeList: \n" + taskUplink2RSUTimeList);
        log.info("taskUplink2CloudTimeList: \n" + taskUplink2CloudTimeList);
        log.info("taskCostTimeList: \n" + taskCostTimeList);
        log.info("taskUssList: \n" + taskUssList);

    }

    @Test
    public void testBinaryRHGPCA_NUMS() {

        log.info("\n");
        log.error("###### ==== testBinary - RHGPCA - Final === ######");

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
        List<Integer> freqAllocArrInit = new ArrayList<>();
        for (int i = 0; i < taskSizeFromDB; i++) {
            // unloadDecisionArrInit.add(new Random().nextInt(vehicleSizeFromDB) - 1);
            unloadDecisionArrInit.add(taskList.get(i).getVehicleID());

            // NOTE: 根据 task 的 cluster_class 参数初始化
            int currTaskClusterID = taskList.get(i).getClusterID();  // 任务聚类 -> class_
            Random random = new Random();
            double ratio = random.nextDouble();
            int tempFreqDefault = (int) (ratio * TaskPolicy.TASK_FREQ_INIT_CLASS_DEFAULT);
            int tempFreqClass = 0;
            if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_0) {
                // tempFreqClass = (int) ((1.0 - ratio) * Constants.TASK_FREQ_INIT_CLASS_0);
                freqAllocArrInit.add(TaskPolicy.TASK_FREQ_INIT_CLASS_0);
                freqAllocArrInit.add(tempFreqDefault + tempFreqClass);
            } else if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_1) {
                // tempFreqClass = (int) ((1.0 - ratio) * Constants.TASK_FREQ_INIT_CLASS_1);
                freqAllocArrInit.add(TaskPolicy.TASK_FREQ_INIT_CLASS_1);
                freqAllocArrInit.add(tempFreqDefault + tempFreqClass);
            } else if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_2) {
                // tempFreqClass = (int) ((1.0 - ratio) * Constants.TASK_FREQ_INIT_CLASS_2);
                freqAllocArrInit.add(TaskPolicy.TASK_FREQ_INIT_CLASS_2);
                freqAllocArrInit.add(tempFreqDefault + tempFreqClass);
            } else if (currTaskClusterID == TaskPolicy.TASK_CLUSTER_CLASS_3) {
                freqAllocArrInit.add(TaskPolicy.TASK_FREQ_INIT_CLASS_3);
                // tempFreqClass = (int) ((1.0 - ratio) * Constants.TASK_FREQ_INIT_CLASS_3);
                freqAllocArrInit.add(tempFreqDefault + tempFreqClass);
            } else {
                freqAllocArrInit.add(new Random().nextInt(TaskPolicy.TASK_FREQ_INIT_CLASS_DEFAULT) + 50);
            }

            freqAllocArr.add(freqAllocArrInit.get(i));
        }

        // 大循环迭代次数
        int maxRound = Constants.NUM_ROUND_TIMES;

        // GA 参数
        int populationSize = Constants.NUM_CHROMOSOMES; // 种群大小
        int numGenerations = Constants.NUM_ITERATIONS; // 迭代代数
        double crossoverRate = 0.8; // 交叉概率
        double mutationRate = 0.01; // 变异概率
        int geneLength = taskSizeFromDB; // 染色体长度(维度数)
        int bound = vehicleSizeFromDB;

        // PSO 参数
        int numParticles = Constants.NUM_PARTICLES; // 粒子数量
        int numDimensions = taskSizeFromDB; // 问题的维度，根据实际问题设置
        int numIterations = Constants.NUM_ITERATIONS; // 迭代次数
        double inertiaWeight = 0.5; // 惯性权重
        double cognitiveWeight = 1.5; // 个体认知权重
        double socialWeight = 1.5; // 群体社会权重
        // double[][] bounds = {{100, 500}, {100, 500}};
        double[][] bounds = new double[numDimensions][2];

        // note：freqRemain ==> bounds[i][1] = freqRemain[unloadArr.get(i) + 1]
        int[] freqRemain = new int[vehicleSizeFromDB + 2];
        freqRemain[0] = (int) cloud.getFreqRemain();
        freqRemain[1] = (int) rsu.getFreqRemain();
        for (int i = 0; i < vehicleSizeFromDB; i++) {
            freqRemain[i + 2] = (int) vehicleList.get(i).getFreqRemain();
        }
        log.info("Init Freq Remain: \n" + Arrays.toString(freqRemain));


        // 记录目标list
        List<Double> uss_avg_list = new ArrayList<>();
        List<Double> energy_avg_list = new ArrayList<>();

        boolean flag_off = false;
        while (!flag_off) {
            // round start ...
            List<int[]> paretoUnloadList = new ArrayList<>();
            List<double[]> paretoFrontList = new ArrayList<>();
            List<double[]> paretoFrontPosList = new ArrayList<>();

            for (int round = 0; round < maxRound; round++) {
                // 卸载决策
                GA_04 ga = new GA_04(populationSize, crossoverRate, mutationRate, geneLength,
                        bound - 1, unloadDecisionArrInit, freqAllocArrInit);
                ga.optimizeChromosomes(numGenerations); // 优化

                // TODO：USSF 选择策略
                int sel_index = 0;
                int gaParetoSize = ga.getParetoFront().size();
                double[] ussf_ga = new double[gaParetoSize];
                //
                double ussMax = getMaxValue(ga.getParetoFront(), 0);
                double ussMin = getMinValue(ga.getParetoFront(), 0);
                double ussDelta = ussMax - ussMin;
                double energyMax = getMaxValue(ga.getParetoFront(), 1);
                double energyMin = getMinValue(ga.getParetoFront(), 1);
                double energDelta = energyMax - energyMin;
                double xi = 0.7;
                double ussf_sum = 0;
                for (int i = 0; i < gaParetoSize; i++) {
                    ussf_ga[i] = xi * (ga.getParetoFront().get(i)[0] - ussMin) / ussDelta
                            + (1 - xi) * (ga.getParetoFront().get(i)[1] - energyMin) / energDelta;
                    ussf_sum += ussf_ga[i];
                }
                double rand = new Random().nextDouble();
                rand = rand * ussf_sum;
                double ussf_sel = 0;
                for (int i = 0; i < gaParetoSize; i++) {
                    ussf_sel += ussf_ga[i];
                    if (ussf_sel >= rand)  {
                        sel_index = i;
                        break;
                    }
                }
                int[] geneArr = ga.getParetoFrontGene().get(sel_index);
                unloadArr.clear();
                for (int i = 0; i < geneArr.length; i++) {
                    unloadArr.add(geneArr[i]);
                }

                for (int i = 0; i < numDimensions; i++) {
                    bounds[i][0] = Constants.MIN_POS_PARTICLE;
                    // bounds[i][1] = 500;
                    bounds[i][1] = Math.min(Constants.MAX_POS_PARTICLE, freqRemain[unloadArr.get(i) + 1]);
                }

                // 资源分配
                PSO_06 p = new PSO_06(numParticles, numDimensions, bounds,
                        inertiaWeight, cognitiveWeight, socialWeight,
                        unloadArr, freqAllocArr);
                p.optimize(numIterations);

                ussMax = getMaxValue(p.getParetoFront(), 0);
                ussMin = getMinValue(p.getParetoFront(), 0);
                ussDelta = ussMax - ussMin;
                energyMax = getMaxValue(p.getParetoFront(), 1);
                energyMin = getMinValue(p.getParetoFront(), 1);
                energDelta = energyMax - energyMin;
                ussf_sum = 0;
                for (int i = 0; i < gaParetoSize; i++) {
                    ussf_ga[i] = xi * (p.getParetoFront().get(i)[0] - ussMin) / ussDelta
                            + (1 - xi) * (p.getParetoFront().get(i)[1] - energyMin) / energDelta;
                    ussf_sum += ussf_ga[i];
                }
                rand = new Random().nextDouble();
                rand = rand * ussf_sum;
                ussf_sel = 0;
                for (int i = 0; i < gaParetoSize; i++) {
                    ussf_sel += ussf_ga[i];
                    if (ussf_sel >= rand)  {
                        sel_index = i;
                        break;
                    }
                }
                // TODO: 可考虑随机取 or 权重取
                int[] unloadGetOne = p.getParetoUnloadArr().get(sel_index);
                double[] freqGetOne = p.getParetoFrontPos().get(sel_index);

                // 修改
                // unloadDecisionArrInit.clear();
                // freqAllocArrInit.clear();
                freqAllocArr.clear();
                for (int i = 0; i < numDimensions; i++) {
                    // unloadDecisionArrInit.add(unloadGetOne[i]);
                    // freqAllocArrInit.add((int) freqGetOne[i]);
                    freqAllocArr.add((int) freqGetOne[i]);
                }

                // 记录
                if (round == maxRound - 1) {
                    paretoUnloadList = p.getParetoUnloadArr();
                    paretoFrontList = p.getParetoFront();
                    paretoFrontPosList = p.getParetoFrontPos();
                }
                // round over !!!
            }

            double uss_avg_round = 0.0;
            double energy_avg_round = 0.0;
            int pareto_size = paretoFrontList.size();

            for (double[] solution : paretoFrontList) {
                uss_avg_round += solution[0] / pareto_size;
                energy_avg_round += solution[1] / pareto_size;
            }

            log.warn("ParetoFrontPos().size() : " + pareto_size);
            for (double[] solution : paretoFrontList) {
                uss_avg_list.add(solution[0]);
                energy_avg_list.add(-solution[1]);
            }

            // uss_avg_list.add(uss_avg_round);
            // energy_avg_list.add(-energy_avg_round);

            if (uss_avg_list.size() >= 200) {
                flag_off = true;
            }
        }

        log.warn("★★★★★ 找到的 Pareto 最优解: ★★★★★");
        uss_avg_list = FormatData.getEffectiveValueList4Digit(uss_avg_list, 3);
        energy_avg_list = FormatData.getEffectiveValueList4Digit(energy_avg_list, 3);
        log.warn("USS avg val: ");
        log.info(uss_avg_list.toString());
        log.warn("Energy avg val: ");
        log.info(energy_avg_list.toString());

        log.warn("******************** task cost time *******************");
        List<Double> taskCostTimeList = new ArrayList<>();
        List<Double> taskUplink2RSUTimeList = new ArrayList<>();
        List<Double> taskUplink2CloudTimeList = new ArrayList<>();
        List<Double> taskUssList = new ArrayList<>();
        Formula.calculateTaskCostTime(taskList, unloadArr, freqAllocArr, taskCostTimeList);
        Formula.calculateTaskTransTime4Uplink2RSU(taskList, unloadArr, taskUplink2RSUTimeList);
        Formula.calculateTaskTransTime4UplinkR2C(taskList, unloadArr, taskUplink2CloudTimeList);
        taskUssList = Formula.getUSS4TaskList(taskList, unloadArr, freqAllocArr);
        for (int i = 0; i < taskSizeFromDB; i++) {
            taskUplink2RSUTimeList.set(i,
                    FormatData.getEffectiveValue4Digit(taskUplink2RSUTimeList.get(i) * 100.0, 2));
            taskUplink2CloudTimeList.set(i,
                    FormatData.getEffectiveValue4Digit(taskUplink2CloudTimeList.get(i) * 100.0, 2));
            taskCostTimeList.set(i,
                    FormatData.getEffectiveValue4Digit(taskCostTimeList.get(i) * 100.0, 2));
            taskUssList.set(i,
                    FormatData.getEffectiveValue4Digit(taskUssList.get(i), 2));
        }

        log.info("taskUplink2RSUTimeList: \n" + taskUplink2RSUTimeList);
        log.info("taskUplink2CloudTimeList: \n" + taskUplink2CloudTimeList);
        log.info("taskCostTimeList: \n" + taskCostTimeList);
        log.info("taskUssList: \n" + taskUssList);

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


