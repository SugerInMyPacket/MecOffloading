import config.InitFrame;
import config.RevisePolicy;
import entity.Cloud;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import resource_allocation.PSO_02;
import resource_allocation.PSO_03;
import resource_allocation.PSO_04;
import resource_allocation.PSO_05;
import unload_decision.GA_01;
import unload_decision.GA_02;
import unload_decision.GA_03;
import unload_decision.GA_04;
import utils.CheckResourceLimit;
import utils.FormatData;
import utils.Formula;
import utils.FormulaLocal;

import java.util.*;
import java.util.function.DoubleToIntFunction;


@Slf4j
public class TestModule {

    static List<Task> taskList = new ArrayList<>();

    static RoadsideUnit rsu = new RoadsideUnit();

    static Cloud cloud = new Cloud();

    // 车辆数目
    static int vehicleNums = 5;
    static List<Vehicle> vehicleList = new ArrayList<>();


    static int len = 10;

    static List<Integer> unloadArr = new ArrayList<>();
    static List<Integer> freqAllocArr = new ArrayList<>();

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
            freqAllocArr.add(100);
        }
    }

    /**
     * @Data 2024-01-03
     */
    @Test
    public void testFormulaTimeExec() {
        initResources();
        initTaskList();
        initLamdaVar();
        // 测试任务集执行时间
        List<Double> taskExecTimeList = new ArrayList<>();
        Map<Integer, Double> taskExecTimeMap = new HashMap<>();
        FormulaLocal.calculateTaskExecTimeLocalOnly(taskList, vehicleList, unloadArr, freqAllocArr, taskExecTimeMap);
        Formula.calculateTaskExecTime(taskList, unloadArr, freqAllocArr, taskExecTimeList);
        for (int i = 0; i < len; i++) {
            log.info("++++++++++++++++" + taskExecTimeList.get(i) + "++++++++++++++++");
        }
    }

    @Test
    public void testFormulaTimeExecLocalOnly() {
        initResources();
        initTaskList();
        initLamdaVar();
        // 测试任务集执行时间
        Map<Integer, Double> taskExecTimeMap = new HashMap<>();
        FormulaLocal.calculateTaskExecTimeLocalOnly(taskList, vehicleList, unloadArr, freqAllocArr,
                taskExecTimeMap);

        for (int i = 0; i < len; i++) {
            log.info("taskID_" + i  + "*** task_VehID_" + taskList.get(i).getVehicleID()
                    + "*** taskExecTime: " + taskExecTimeMap.get(i) + "++");
        }
    }

    /**
     * @Data 2024-01-03
     */
    @Test
    public void testFormulaTimeUplinkTime2RSU() {
        initResources();
        initTaskList();
        initLamdaVar();
        // 测试任务集上传时间
        List<Double> taskUplinkTimeList2RSU = new ArrayList<>();
        Formula.calculateTaskTransTime4Uplink2RSU(taskList, unloadArr, taskUplinkTimeList2RSU);
        for (int i = 0; i < len; i++) {
            log.info("++++++++++++++++" + taskUplinkTimeList2RSU.get(i) + "++++++++++++++++");
        }
    }

    /**
     * @Data 2024-01-09
     */
    @Test
    public void testFromulaUSS() {
        initResources();
        initTaskList();
        initLamdaVar();

        System.out.println("卸载决策: " + Arrays.toString(unloadArr.toArray()));
        System.out.println("资源分配: " + Arrays.toString(freqAllocArr.toArray()));

        // 初始化车辆信道增益
        List<Double> gainChannelVehicleList = new ArrayList<>();
        for (int i = 0; i < vehicleNums; i++) {
            gainChannelVehicleList.add(new Random().nextDouble());
        }
        Formula.initGainChannelVehicles(gainChannelVehicleList);

        // 测试任务集满意度计算
        List<Double> taskUSSList = new ArrayList<>();
        List<Double> taskCostTimeList = new ArrayList<>();
        // for (int i = 0; i < len; i++) taskCostTimeList.add(new Random().nextDouble());

        Formula.calculateTaskCostTime(taskList, unloadArr, freqAllocArr, taskCostTimeList);
        for (int i = 0; i < len; i++) {
            log.info("----------" + taskCostTimeList.get(i) + "----------");
        }

        Formula.getSatisfaction4TaskList(taskList, taskCostTimeList, taskUSSList);
        for (int i = 0; i < len; i++) {
            log.info("++++++" + taskUSSList.get(i) + "++++++");
        }


        List<Double> taskCostEnergyList = new ArrayList<>();
        Formula.calculateEnergy4TaskList(taskList, unloadArr, freqAllocArr, taskCostEnergyList);
        for (int i = 0; i < len; i++) {
            log.info("++++++" + taskCostEnergyList.get(i) + "++++++");
        }

    }

    /**
     * @Data 2024-01-10
     */
    @Test
    public void testGA_PSO_USS_Energy() {

        initResources();
        initTaskList();
        // initLamdaVar();

        log.info("TestSchedule ===> 测试 GA algorithm .......");
        int populationSize = 20; // 种群大小
        double crossoverRate = 0.8; // 交叉概率
        double mutationRate = 0.01; // 变异概率
        int geneLength = len; // 染色体长度(维度数)
        int numGenerations = 50; // 迭代代数

        List<Integer> unloadDecisionArr = new ArrayList<>(); // 个数应该为 geneLength
        for (int i = 0; i < geneLength; i++) {
            unloadDecisionArr.add(new Random().nextInt(vehicleNums) - 1);
        }
        GA_01 ga = new GA_01(populationSize, crossoverRate, mutationRate, geneLength);
        ga.initPopulation(unloadDecisionArr); // 初始化
        ga.optimizeChromosomes(numGenerations); // 优化
        log.info("GA 当前最优染色体: " + Arrays.toString(ga.getCurrBestChromosome().getGene()));

        for (int i = 0; i < len; i++) {
            unloadArr.add(ga.getCurrBestChromosome().getGene()[i]);
        }
        // System.out.println("卸载决策: " + Arrays.toString(unloadArr.toArray()));


        log.info("TestSchedule ===> 测试PSO算法........");
        PSO_02 p = new PSO_02();
        int numParticles = 30; // 粒子数量
        int numDimensions = len; // 问题的维度，根据实际问题设置
        int numIterations = 100; // 迭代次数
        double inertiaWeight = 0.5; // 惯性权重
        double cognitiveWeight = 1.5; // 个体认知权重
        double socialWeight = 1.5; // 群体社会权重
        List<Integer> resArr = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            resArr.add(new Random().nextInt(100) + 1);
        }
        // 初始化粒子
        p.initParticles(numParticles, numDimensions, resArr);
        // 优化
        p.optimizeParticles(numIterations, inertiaWeight, cognitiveWeight, socialWeight, 1, 100, -1, -1);
        log.info("PSO 当前最优粒子: " + Arrays.toString(p.getBestParticle().getPos()));

        for (int i = 0; i < len; i++) {
            freqAllocArr.add((int) p.getBestParticle().getPos()[i]);
        }

        log.error(".............. 最终输出结果如下 ...................");
        log.info("★★★ 卸载决策: " + Arrays.toString(unloadArr.toArray()));
        log.info("★★★ 资源分配: " + Arrays.toString(freqAllocArr.toArray()));


        // ------------------ 初始化车辆信道增益
        List<Double> gainChannelVehicleList = new ArrayList<>();
        for (int i = 0; i < vehicleNums; i++) {
            gainChannelVehicleList.add(new Random().nextDouble());
        }
        Formula.initGainChannelVehicles(gainChannelVehicleList);


        // -------------------- 测试任务集满意度计算
        List<Double> taskUSSList = new ArrayList<>();
        List<Double> taskCostTimeList = new ArrayList<>();

        // 任务花费时间计算
        Formula.calculateTaskCostTime(taskList, unloadArr, freqAllocArr, taskCostTimeList);
        for (int i = 0; i < len; i++) {
            log.info("-------" + taskCostTimeList.get(i) + "-------");
        }

        // 任务满意度计算
        Formula.getSatisfaction4TaskList(taskList, taskCostTimeList, taskUSSList);
        for (int i = 0; i < len; i++) {
            log.info("++++++" + taskUSSList.get(i) + "++++++");
        }

        // 任务能耗计算
        List<Double> taskCostEnergyList = new ArrayList<>();
        Formula.calculateEnergy4TaskList(taskList, unloadArr, freqAllocArr, taskCostEnergyList);
        for (int i = 0; i < len; i++) {
            log.info("++++++" + taskCostEnergyList.get(i) + "++++++");
        }


    }

    /**
     * @Data 2024-01-14
     */
    @Test
    public void testCheckAndRevise() {

        initResources();
        initTaskList();
        initLamdaVar();

        boolean[] vehicleAllocResIsLegal = new boolean[vehicleNums];
        Map<Integer, List<Integer>> vehicleResNotEnoughMap = new HashMap<>();

        // 检查车辆资源是否足够
        CheckResourceLimit.checkVehicleLimit(vehicleList, taskList, unloadArr, freqAllocArr, vehicleAllocResIsLegal, vehicleResNotEnoughMap);

        log.info("★★★ 卸载决策: " + Arrays.toString(unloadArr.toArray()));
        log.info("★★★ 资源分配: " + Arrays.toString(freqAllocArr.toArray()));

        log.info("=====> vehicleAllocResIsLegal[] : " + Arrays.toString(vehicleAllocResIsLegal));


        // 初始化车辆信道增益
        List<Double> gainChannelVehicleList = new ArrayList<>();
        for (int i = 0; i < vehicleNums; i++) {
            gainChannelVehicleList.add(new Random().nextDouble());
        }
        Formula.initGainChannelVehicles(gainChannelVehicleList);

        // 测试任务集满意度计算
        List<Double> taskUSSList = new ArrayList<>();
        List<Double> taskCostTimeList = new ArrayList<>();
        // for (int i = 0; i < len; i++) taskCostTimeList.add(new Random().nextDouble());

        Formula.calculateTaskCostTime(taskList, unloadArr, freqAllocArr, taskCostTimeList);
        for (int i = 0; i < len; i++) {
            log.info("----------" + taskCostTimeList.get(i) + "----------");
        }

        Formula.getSatisfaction4TaskList(taskList, taskCostTimeList, taskUSSList);
        for (int i = 0; i < len; i++) {
            log.info("++++++" + taskUSSList.get(i) + "++++++");
        }

        // 任务能耗计算
        List<Double> taskCostEnergyList = new ArrayList<>();
        Formula.calculateEnergy4TaskList(taskList, unloadArr, freqAllocArr, taskCostEnergyList);
        for (int i = 0; i < len; i++) {
            log.info("++++++" + taskCostEnergyList.get(i) + "++++++");
        }

        // 修正
        RevisePolicy.reviseUnloadArr(taskList, vehicleList, rsu, unloadArr, freqAllocArr, taskUSSList, taskCostEnergyList);
        log.warn("--------------------- 修正后 ---------------------");
        log.info("★★★ 卸载决策: " + Arrays.toString(unloadArr.toArray()));
        log.info("★★★ 资源分配: " + Arrays.toString(freqAllocArr.toArray()));

    }

    /**
    * @Data 2024-01-16
    */
    @Test
    public void testAll() {
        // 初始化
        // InitFrame.init();
        InitFrame.initFromDB();
        // InitFrame.initTaskList();
        // InitFrame.initAllRes();

        taskList = InitFrame.getTaskList();
        vehicleList = InitFrame.getVehicleList();
        rsu = InitFrame.getRSU();


        taskSizeFromDB = taskList.size();
        System.out.println("taskSizeFromDB: " + taskSizeFromDB);
        vehicleSizeFromDB = vehicleList.size();

        // initLamdaVar();

        log.info("TestSchedule ===> 测试 GA algorithm .......");
        int populationSize = 20; // 种群大小
        double crossoverRate = 0.8; // 交叉概率
        double mutationRate = 0.01; // 变异概率
        int geneLength = taskSizeFromDB; // 染色体长度(维度数)
        int numGenerations = 50; // 迭代代数

        List<Integer> unloadDecisionArr = new ArrayList<>(); // 个数应该为 geneLength
        List<Integer> freqAllocArrInit = new ArrayList<>();
        for (int i = 0; i < geneLength; i++) {
            unloadDecisionArr.add(new Random().nextInt(vehicleSizeFromDB) - 1);
            freqAllocArrInit.add(new Random().nextInt(100) + 50);
        }
        // System.out.println("unloadDecisionArr:" + unloadDecisionArr);

        // GA_01 ga = new GA_01(populationSize, crossoverRate, mutationRate, geneLength);
        GA_02 ga = new GA_02(populationSize, crossoverRate, mutationRate, geneLength, freqAllocArrInit);
        ga.initPopulation(unloadDecisionArr); // 初始化
        ga.optimizeChromosomes(numGenerations); // 优化
        log.info("GA 当前最优染色体: " + Arrays.toString(ga.getCurrBestChromosome().getGene()));

        for (int i = 0; i < taskSizeFromDB; i++) {
            unloadArr.add(ga.getCurrBestChromosome().getGene()[i]);
        }

        log.info("TestSchedule ===> 测试PSO算法........");
        // PSO_02 p = new PSO_02();
        int numParticles = 30; // 粒子数量
        int numDimensions = taskSizeFromDB; // 问题的维度，根据实际问题设置
        int numIterations = 100; // 迭代次数
        double inertiaWeight = 0.5; // 惯性权重
        double cognitiveWeight = 1.5; // 个体认知权重
        double socialWeight = 1.5; // 群体社会权重
        List<Integer> resArr = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            resArr.add(new Random().nextInt(100) + 1);
        }

        PSO_03 p = new PSO_03(numParticles, numDimensions, unloadArr);
        // 初始化粒子
        p.initParticles(numParticles, numDimensions, resArr);
        // 优化
        p.optimizeParticles(numIterations, inertiaWeight, cognitiveWeight, socialWeight, 10, 500, -1, -1);
        log.info("PSO 当前最优粒子: " + Arrays.toString(p.getBestParticle().getPos()));

        for (int i = 0; i < taskSizeFromDB; i++) {
            freqAllocArr.add((int) p.getBestParticle().getPos()[i]);
        }

        log.warn(".............. 最终输出结果如下 ...................");
        log.info("★★★ 卸载决策: " + Arrays.toString(unloadArr.toArray()));
        log.info("★★★ 资源分配: " + Arrays.toString(freqAllocArr.toArray()));

    }


    /**
    * @Data 2024-02-21
    */
    @Test
    public void testParetoExample() {

        InitFrame.init();

        log.info("★★★ 卸载决策: " + Arrays.toString(unloadArr.toArray()));
        log.info("★★★ 资源分配: " + Arrays.toString(freqAllocArr.toArray()));

        // 初始化车辆信道增益
        List<Double> gainChannelVehicleList = new ArrayList<>();
        for (int i = 0; i < vehicleNums; i++) {
            gainChannelVehicleList.add(new Random().nextDouble());
        }
        Formula.initGainChannelVehicles(gainChannelVehicleList);


        log.info("TestSchedule ===> 测试 GA algorithm .......");
        int populationSize = 20; // 种群大小
        double crossoverRate = 0.8; // 交叉概率
        double mutationRate = 0.01; // 变异概率
        int geneLength = len; // 染色体长度(维度数)
        int numGenerations = 50; // 迭代代数

        int bound = vehicleNums;

        List<Integer> unloadDecisionArrInit = new ArrayList<>(); // 个数应该为 geneLength
        List<Integer> freqAllocArrInit = new ArrayList<>();
        for (int i = 0; i < geneLength; i++) {
            unloadDecisionArrInit.add(new Random().nextInt(bound) - 1);
            freqAllocArrInit.add(new Random().nextInt(100) + 50);
        }

        GA_03 ga = new GA_03(populationSize, crossoverRate, mutationRate,
                geneLength, bound - 1, freqAllocArrInit);
        // ga.initPopulation(unloadDecisionArrInit); // 初始化
        ga.optimizeChromosomes(numGenerations); // 优化

        for (int[] res : ga.getParetoFrontGene()) {
            System.out.println("unloadArr:" + Arrays.toString(res));
        }

        log.info("TestSchedule ===> 测试PSO算法........");
        int numParticles = 30; // 粒子数量
        int numDimensions = len; // 问题的维度，根据实际问题设置
        int numIterations = 100; // 迭代次数
        double inertiaWeight = 0.5; // 惯性权重
        double cognitiveWeight = 1.5; // 个体认知权重
        double socialWeight = 1.5; // 群体社会权重
        // double[][] bounds = {{10, 500}, {10, 500}};
        double[][] bounds = new double[numDimensions][2];
        for (int i = 0; i < numDimensions; i++) {
            bounds[i][0] = 10;
            bounds[i][1] = 500;
        }
        List<Integer> resArr = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            resArr.add(new Random().nextInt(100) + 1);
        }

        PSO_04 p = new PSO_04(numParticles, numDimensions, bounds, inertiaWeight, cognitiveWeight,
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
            System.out.println("resAlloc:" + Arrays.toString(array));
        }

    }

    /**
    * @Data 2024-02-21
    */
    @Test
    public void testPareto() {
        // 初始化
        InitFrame.initFromDB();

        taskList = InitFrame.getTaskList();
        vehicleList = InitFrame.getVehicleList();
        rsu = InitFrame.getRSU();


        taskSizeFromDB = taskList.size();
        System.out.println("taskSizeFromDB: " + taskSizeFromDB);
        vehicleSizeFromDB = vehicleList.size();

        // initLamdaVar();

        log.info("TestSchedule ===> 测试 GA algorithm .......");
        int populationSize = 20; // 种群大小
        double crossoverRate = 0.8; // 交叉概率
        double mutationRate = 0.01; // 变异概率
        int geneLength = taskSizeFromDB; // 染色体长度(维度数)
        int numGenerations = 50; // 迭代代数

        List<Integer> unloadDecisionArrInit = new ArrayList<>(); // 个数应该为 geneLength
        List<Integer> freqAllocArrInit = new ArrayList<>();
        for (int i = 0; i < geneLength; i++) {
            unloadDecisionArrInit.add(new Random().nextInt(vehicleSizeFromDB) - 1);
            freqAllocArrInit.add(new Random().nextInt(100) + 50);
        }

        GA_02 ga = new GA_02(populationSize, crossoverRate, mutationRate, geneLength, freqAllocArrInit);
        ga.initPopulation(unloadDecisionArrInit); // 初始化
        ga.optimizeChromosomes(numGenerations); // 优化
        log.info("GA 当前最优染色体: " + Arrays.toString(ga.getCurrBestChromosome().getGene()));

        for (int i = 0; i < taskSizeFromDB; i++) {
            unloadArr.add(ga.getCurrBestChromosome().getGene()[i]);
        }

        log.info("TestSchedule ===> 测试PSO算法........");
        int numParticles = 30; // 粒子数量
        int numDimensions = taskSizeFromDB; // 问题的维度，根据实际问题设置
        int numIterations = 100; // 迭代次数
        double inertiaWeight = 0.5; // 惯性权重
        double cognitiveWeight = 1.5; // 个体认知权重
        double socialWeight = 1.5; // 群体社会权重
        // double[][] bounds = {{10, 500}, {10, 500}};
        double[][] bounds = new double[numDimensions][2];
        for (int i = 0; i < numDimensions; i++) {
            bounds[i][0] = 10;
            bounds[i][1] = 500;
        }
        List<Integer> resArr = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            resArr.add(new Random().nextInt(100) + 1);
        }

        PSO_04 p = new PSO_04(numParticles, numDimensions, bounds, inertiaWeight, cognitiveWeight,
                socialWeight, unloadArr);
        p.optimize(numIterations);

        System.out.println(unloadArr);
        for (double[] res : p.getParetoFrontPos()) {
            int[] array = Arrays.stream(res).mapToInt(new DoubleToIntFunction() {
                @Override
                public int applyAsInt(double value) {
                    return (int) value;
                }
            }).toArray();
            System.out.println("resAlloc:" + Arrays.toString(array));
        }

    }

    /**
    * @Data 2024-02-21
    */
    @Test
    public void testBinary01() {
        // 初始化
        InitFrame.initFromDB();

        taskList = InitFrame.getTaskList();
        vehicleList = InitFrame.getVehicleList();
        rsu = InitFrame.getRSU();

        taskSizeFromDB = taskList.size();
        System.out.println("taskSizeFromDB: " + taskSizeFromDB);
        vehicleSizeFromDB = vehicleList.size();

        // initLamdaVar();

        log.info("TestSchedule ===> 测试 GA algorithm .......");
        int populationSize = 20; // 种群大小
        double crossoverRate = 0.8; // 交叉概率
        double mutationRate = 0.01; // 变异概率
        int geneLength = taskSizeFromDB; // 染色体长度(维度数)
        int numGenerations = 50; // 迭代代数
        int bound = vehicleSizeFromDB;

        List<Integer> unloadDecisionArrInit = new ArrayList<>(); // 个数应该为 geneLength
        List<Integer> freqAllocArrInit = new ArrayList<>();
        for (int i = 0; i < geneLength; i++) {
            unloadDecisionArrInit.add(new Random().nextInt(vehicleSizeFromDB) - 1);
            freqAllocArrInit.add(new Random().nextInt(200) + 50);
        }

        GA_03 ga = new GA_03(populationSize, crossoverRate, mutationRate, geneLength, bound - 1, freqAllocArrInit);
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

        log.info("TestSchedule ===> 测试PSO算法........");
        int numParticles = 20; // 粒子数量
        int numDimensions = taskSizeFromDB; // 问题的维度，根据实际问题设置
        int numIterations = 50; // 迭代次数
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

        for (int i = 0; i < numDimensions; i++) {
            bounds[i][0] = 10;
            // bounds[i][1] = 500;
            bounds[i][1] = freqRemain[unloadArr.get(i)];
        }

        PSO_04 p = new PSO_04(numParticles, numDimensions, bounds, inertiaWeight, cognitiveWeight,
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
            System.out.println("resAlloc:" + Arrays.toString(array));
            break;
        }
    }

    /**
     * @Data 2024-02-21
     */
    @Test
    public void testBinary02() {
        log.warn("###################### ==== testBinary02() === ######################");
        // 初始化
        InitFrame.initFromDB();

        taskList = InitFrame.getTaskList();
        vehicleList = InitFrame.getVehicleList();
        rsu = InitFrame.getRSU();
        cloud = InitFrame.getCloud();

        taskSizeFromDB = taskList.size();
        System.out.println("taskSizeFromDB: " + taskSizeFromDB);
        vehicleSizeFromDB = vehicleList.size();

        // initLamdaVar();

        int maxRound = 10;

        log.info("TestSchedule ===> 测试 GA algorithm .......");
        int populationSize = 20; // 种群大小
        double crossoverRate = 0.8; // 交叉概率
        double mutationRate = 0.01; // 变异概率
        int geneLength = taskSizeFromDB; // 染色体长度(维度数)
        int numGenerations = 50; // 迭代代数
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

        log.info("TestSchedule ===> 测试PSO算法........");
        int numParticles = 20; // 粒子数量
        int numDimensions = taskSizeFromDB; // 问题的维度，根据实际问题设置
        int numIterations = 100; // 迭代次数
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
            bounds[i][0] = 10;
            // bounds[i][1] = 500;
            bounds[i][1] = Math.min(1000, freqRemain[unloadArr.get(i) + 1]);
        }

        PSO_05 p = new PSO_05(numParticles, numDimensions, bounds, inertiaWeight, cognitiveWeight,
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

        log.info("★★★★★ 找到的Pareto最优解: ★★★★★");
        for (double[] solution : p.getParetoFront()) {
            log.info("USS val: " + solution[0] + ", Energy val: " + -solution[1]);
        }

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
        List<Double> taskCostTimeList = new ArrayList<>();
        List<Double> taskUssList = new ArrayList<>();
        Formula.calculateTaskCostTime(taskList, unloadArr, freqAllocArr, taskCostTimeList);
        taskUssList = Formula.getUSS4TaskList(taskList, unloadArr, freqAllocArr);
        for (int i = 0; i < taskSizeFromDB; i++) {
            taskCostTimeList.set(i, FormatData.getEffectiveValue4Digit(taskCostTimeList.get(i), 2));
            taskUssList.set(i, FormatData.getEffectiveValue4Digit(taskUssList.get(i), 2));
        }

        log.info("taskCostTimeList: \n" + taskCostTimeList);
        log.info("taskUssList: \n" + taskUssList);
    }


}
