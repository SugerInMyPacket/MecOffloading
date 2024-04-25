import compare.PSOCO;
import config.InitFrame;
import config.RevisePolicy;
import entity.Cloud;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import enums.Constants;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import utils.FormatData;
import utils.Formula;
import utils.FormulaLocal;

import java.util.*;

@Slf4j
public class TestCompare {

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
    * @Data 2024-03-10
    */
    @Test
    public void testLocalOnly() {
        log.error("############### test LocalOnly ###############");
        // 初始化
        InitFrame.initFromDB();
        // 读取信息
        taskList = InitFrame.getTaskList();
        vehicleList = InitFrame.getVehicleList();
        rsu = InitFrame.getRSU();
        cloud = InitFrame.getCloud();

        taskSizeFromDB = taskList.size();
        vehicleSizeFromDB = vehicleList.size();
        log.warn("vehicleSizeFromDB: " + vehicleSizeFromDB);
        log.warn("taskSizeFromDB: " + taskSizeFromDB);
        // 初始化
        InitFrame.initFromDB();

        int round = 200;
        double[] ussAvgArr = new double[round];
        double[] energyAvgArr = new double[round];

        List<Double> ussAvgList = new ArrayList<>();
        List<Double> energyAvgList = new ArrayList<>();

        for (int i = 0; i < round; i++) {
            Random random = new Random();
            unloadArr.clear();
            freqAllocArr.clear();
            for (int j = 0; j < taskSizeFromDB; j++) {
                int vID = taskList.get(j).getVehicleID();
                unloadArr.add(vID);
                // freqAllocArr.add(random.nextInt((int) vehicleList.get(vID - 1).getFreqRemain()) * 3);
                // freqAllocArr.add((int) (random.nextInt((i % 8 + 2) * 50) * Constants.DATA_SIZE_MULTI_INCREASE));
                freqAllocArr.add((int) (random.nextInt(400) * Constants.DATA_SIZE_MULTI_INCREASE));
            }
            // 测试任务集执行时间
            Map<Integer, Double> taskExecTimeMap = new HashMap<>();
            FormulaLocal.calculateTaskExecTimeLocalOnly(taskList, vehicleList,
                    unloadArr, freqAllocArr,
                    taskExecTimeMap);

            // for (int j = 0; j < taskSizeFromDB; j++) {
            //     taskExecTimeMap.put(j,
            //             FormatData.getEffectiveValue4Digit(taskExecTimeMap.get(j) * 100.0, 2));
            // }
            // log.info("Task exec time list local only : " + taskExecTimeMap.values());

            List<Double> taskUSSList =
                    FormulaLocal.getUss4TaskListLocalOnly(taskList, vehicleList, unloadArr, freqAllocArr);
            List<Double> taskEnergyList =
                    FormulaLocal.getEnergy4TaskListLocalOnly(taskList, unloadArr, freqAllocArr);

            taskUSSList = FormatData.getEffectiveValueList4Digit(taskUSSList, 5);
            taskEnergyList = FormatData.getEffectiveValueList4Digit(taskEnergyList, 5);
            double uss_total = 0;
            double energy_total = 0;

            for (int j = 0; j < taskSizeFromDB; j++) {
                uss_total += taskUSSList.get(j);
                energy_total += taskEnergyList.get(j);
            }

            ussAvgArr[i] = uss_total / taskSizeFromDB;
            energyAvgArr[i] = energy_total / taskSizeFromDB / 40;

            ussAvgList.add(ussAvgArr[i]);
            energyAvgList.add(energyAvgArr[i]);
        }

        double uss_total_round = 0;
        double energy_total_round = 0;

        for (int i = 0; i < round; i++) {
            uss_total_round += ussAvgArr[i];
            energy_total_round += energyAvgArr[i];
        }

        log.info("uss_avg: " + FormatData.getEffectiveValue4Digit(uss_total_round / round, 5));
        log.info("energy_avg: " + FormatData.getEffectiveValue4Digit(energy_total_round / round / 100, 5));

        ussAvgList = FormatData.getEffectiveValueList4Digit(ussAvgList, 3);
        energyAvgList = FormatData.getEffectiveValueList4Digit(energyAvgList, 3);
        log.warn("ussAvgList: \n" + ussAvgList);
        log.warn("energyAvgList: \n" + energyAvgList);
        // log.info("taskUSSList: " + taskUSSList);
        // log.info("taskEnergyList: " + taskEnergyList);

    }

    /**
    * @Data 2024-03-07
    */
    @Test
    public void testGreedy() {
        log.info("############### test Greedy ###############");

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

        Random random = new Random();
        for (int i = 0; i < taskSizeFromDB; i++) {
            // unloadArr.add(random.nextInt(vehicleSizeFromDB + 1) - 1);
            unloadArr.add(0);
            // freqAllocArr.add(random.nextInt(400) + 100);
            freqAllocArr.add(200);
        }

        int[] freqRemain = new int[vehicleSizeFromDB + 2];
        freqRemain[0] = (int) cloud.getFreqRemain();
        freqRemain[1] = (int) rsu.getFreqRemain();
        for (int i = 0; i < vehicleSizeFromDB; i++) {
            freqRemain[i + 2] = (int) vehicleList.get(i).getFreqRemain();
        }
        log.info("Init Freq Remain: \n" + Arrays.toString(freqRemain));

        List<Double> taskUssList = new ArrayList<>();
        List<Double> taskEnergyList = new ArrayList<>();
        List<Double> taskCostTimeList = new ArrayList<>();
        List<Double> taskUplinkTime2RSUList = new ArrayList<>();
        Formula.calculateTaskCostTime(taskList, unloadArr, freqAllocArr, taskCostTimeList);
        Formula.calculateTaskTransTime4Uplink2RSU(taskList, unloadArr, taskUplinkTime2RSUList);
        for (int i = 0; i < taskSizeFromDB; i++) {
            taskCostTimeList.set(i, FormatData.getEffectiveValue4Digit(taskCostTimeList.get(i) * 100.0, 3));
            taskUplinkTime2RSUList.set(i, FormatData.getEffectiveValue4Digit(taskUplinkTime2RSUList.get(i) * 100.0, 3));
        }
        // log.info("task cost time : \n" + taskCostTimeList);
        // log.info("task uplink time rsu : \n" + taskUplinkTime2RSUList);
        taskUssList = Formula.getUSS4TaskList(taskList, unloadArr, freqAllocArr);
        taskEnergyList = Formula.getEnergy4TaskList(taskList, unloadArr, freqAllocArr);

        // 贪婪修正
        // RevisePolicy.reviseUnloadArr(taskList, vehicleList, rsu, unloadArr, freqAllocArr, taskUssList, taskEnergyList);
        RevisePolicy.reviseUnloadArr(taskList, vehicleList, rsu, unloadArr, freqAllocArr);

        taskUssList = Formula.getUSS4TaskList(taskList, unloadArr, freqAllocArr);
        taskEnergyList = Formula.getEnergy4TaskList(taskList, unloadArr, freqAllocArr);

        for (int i = 0; i < taskSizeFromDB; i++) {
            freqRemain[unloadArr.get(i) + 1] = freqRemain[unloadArr.get(i) + 1] - freqAllocArr.get(i);
        }

        double taskUssTotal = 0.0;
        double taskEnergyTotal = 0.0;
        for (int i = 0; i < taskSizeFromDB; i++) {
            taskUssTotal += taskUssList.get(i);
            taskEnergyTotal += taskEnergyList.get(i);
        }

        log.info("taskUssAvg: " + FormatData.getEffectiveValue4Digit(taskUssTotal / taskSizeFromDB, 5));
        log.info("taskEnergyAvg: " + FormatData.getEffectiveValue4Digit(taskEnergyTotal / 100 / taskSizeFromDB, 5));

        log.info("freq remain arr : \n" + Arrays.toString(freqRemain));
        log.info("unload arr : " + unloadArr);
        log.info("freq alloc arr : " + freqAllocArr);
    }

    @Test
    public void testGreedy2() {
        log.info("\n");
        log.error("############### test Greedy ###############");

        // 初始化
        InitFrame.initFromDB();

        // 读取信息
        taskList = InitFrame.getTaskList();
        vehicleList = InitFrame.getVehicleList();
        rsu = InitFrame.getRSU();
        cloud = InitFrame.getCloud();

        taskSizeFromDB = taskList.size();
        vehicleSizeFromDB = vehicleList.size();
        log.warn("taskSizeFromDB: " + taskSizeFromDB);
        log.warn("vehicleSizeFromDB: " + vehicleSizeFromDB);


        int[] freqRemain = new int[vehicleSizeFromDB + 2];
        freqRemain[0] = (int) cloud.getFreqRemain();
        freqRemain[1] = (int) rsu.getFreqRemain();
        for (int i = 0; i < vehicleSizeFromDB; i++) {
            freqRemain[i + 2] = (int) vehicleList.get(i).getFreqRemain();
        }
        // log.info("Init Freq Remain: \n" + Arrays.toString(freqRemain));

        List<Double> ussAvgList = new ArrayList<>();
        List<Double> energyAvgList = new ArrayList<>();

        int round = 100;
        double[] ussAvgArr = new double[round];
        double[] energyAvgArr = new double[round];


        for (int i = 0; i < round; i++) {
            // 随机
            Random random = new Random();
            unloadArr.clear();
            freqAllocArr.clear();
            for (int j = 0; j < taskSizeFromDB; j++) {
                unloadArr.add(random.nextInt(vehicleSizeFromDB + 1) - 1);
                // unloadArr.add(taskList.get(j).getVehicleID());
                freqAllocArr.add(random.nextInt(400) + 100);
            }

            List<Double> taskUssList = new ArrayList<>();
            List<Double> taskEnergyList = new ArrayList<>();

            // 贪婪修正
            RevisePolicy.reviseUnloadArr(taskList, vehicleList, rsu, unloadArr, freqAllocArr);

            taskUssList = Formula.getUSS4TaskList(taskList, unloadArr, freqAllocArr);
            taskEnergyList = Formula.getEnergy4TaskList(taskList, unloadArr, freqAllocArr);

            double taskUssTotal = 0.0;
            double taskEnergyTotal = 0.0;
            for (int j = 0; j < taskSizeFromDB; j++) {
                taskUssTotal += taskUssList.get(j);
                taskEnergyTotal += taskEnergyList.get(j) / 1000.0;
            }

            ussAvgArr[i] = taskUssTotal / taskSizeFromDB;
            energyAvgArr[i] = taskEnergyTotal / taskSizeFromDB;

            ussAvgList.add(ussAvgArr[i]);
            energyAvgList.add(energyAvgArr[i]);
        }
        double uss_total_round = 0;
        double energy_total_round = 0;

        for (int i = 0; i < round; i++) {
            uss_total_round += ussAvgArr[i];
            energy_total_round += energyAvgArr[i];
        }

        log.info("uss_avg: " + FormatData.getEffectiveValue4Digit(uss_total_round / round, 3));
        log.info("energy_avg: " + FormatData.getEffectiveValue4Digit(energy_total_round / round, 3));

        ussAvgList = FormatData.getEffectiveValueList4Digit(ussAvgList, 3);
        energyAvgList = FormatData.getEffectiveValueList4Digit(energyAvgList, 3);
        log.warn("ussAvgList: \n" + ussAvgList.toString());
        log.warn("energyAvgList: \n" + energyAvgList.toString());

        List<Double> uplinkTime2RSU = new ArrayList<>();
        List<Double> execTime = new ArrayList<>();
        Formula.calculateTaskTransTime4Uplink2RSU(taskList, unloadArr, uplinkTime2RSU);
        Formula.calculateTaskExecTime(taskList, unloadArr, freqAllocArr, execTime);
        uplinkTime2RSU = FormatData.getEffectiveValueList4Digit(uplinkTime2RSU, 3);
        execTime = FormatData.getEffectiveValueList4Digit(execTime, 3);
        log.info("uplink time 2 rsu: " + uplinkTime2RSU);
        log.info("exec time 2 rsu: " + execTime);
    }

    @Test
    public void testGreedy3() {
        log.info("\n");
        log.error("############### test Greedy ###############");

        // 初始化
        InitFrame.initFromDB();

        // 读取信息
        taskList = InitFrame.getTaskList();
        vehicleList = InitFrame.getVehicleList();
        rsu = InitFrame.getRSU();
        cloud = InitFrame.getCloud();

        taskSizeFromDB = taskList.size();
        vehicleSizeFromDB = vehicleList.size();
        log.warn("taskSizeFromDB: " + taskSizeFromDB);
        log.warn("vehicleSizeFromDB: " + vehicleSizeFromDB);


        int[] freqRemain = new int[vehicleSizeFromDB + 2];
        freqRemain[0] = (int) cloud.getFreqRemain();
        freqRemain[1] = (int) rsu.getFreqRemain();
        for (int i = 0; i < vehicleSizeFromDB; i++) {
            freqRemain[i + 2] = (int) vehicleList.get(i).getFreqRemain();
        }
        // log.info("Init Freq Remain: \n" + Arrays.toString(freqRemain));

        List<Double> ussAvgList = new ArrayList<>();
        List<Double> energyAvgList = new ArrayList<>();

        for (int j = 0; j < 200; j++) {
            int round = 100;
            double[] ussAvgArr = new double[round];
            double[] energyAvgArr = new double[round];


            for (int i = 0; i < round; i++) {
                // 随机
                Random random = new Random();
                unloadArr.clear();
                freqAllocArr.clear();
                for (int k = 0; k < taskSizeFromDB; k++) {
                    double rk = random.nextDouble();
                    if (rk < 0.6) {
                        unloadArr.add(random.nextInt(vehicleSizeFromDB + 1) - 1);
                    } else {
                        unloadArr.add(taskList.get(j).getVehicleID());
                    }
                    // unloadArr.add(taskList.get(j).getVehicleID());
                    freqAllocArr.add(random.nextInt((j % 8 + 2) * 50) + 100);
                }

                List<Double> taskUssList = new ArrayList<>();
                List<Double> taskEnergyList = new ArrayList<>();

                // 贪婪修正
                RevisePolicy.reviseUnloadArr(taskList, vehicleList, rsu, unloadArr, freqAllocArr);

                taskUssList = Formula.getUSS4TaskList(taskList, unloadArr, freqAllocArr);
                taskEnergyList = Formula.getEnergy4TaskList(taskList, unloadArr, freqAllocArr);

                double taskUssTotal = 0.0;
                double taskEnergyTotal = 0.0;
                for (int z = 0; z < taskSizeFromDB; z++) {
                    taskUssTotal += taskUssList.get(z);
                    taskEnergyTotal += taskEnergyList.get(z) / 1000.0;
                }

                ussAvgArr[i] = taskUssTotal / taskSizeFromDB;
                energyAvgArr[i] = taskEnergyTotal / taskSizeFromDB;
            }
            double uss_total_round = 0;
            double energy_total_round = 0;

            for (int i = 0; i < round; i++) {
                uss_total_round += ussAvgArr[i];
                energy_total_round += energyAvgArr[i];
            }

            ussAvgList.add(uss_total_round / round);
            energyAvgList.add(energy_total_round / round + 100);

        }

        ussAvgList = FormatData.getEffectiveValueList4Digit(ussAvgList, 3);
        energyAvgList = FormatData.getEffectiveValueList4Digit(energyAvgList, 3);
        log.warn("ussAvgList: \n" + ussAvgList.toString());
        log.warn("energyAvgList: \n" + energyAvgList.toString());
    }


    /**
    * @Data 2024-03-12
    */
    @Test
    public void testPSOCO() {
        log.info("\n");
        log.error(" ===> 测试PSOCO算法........");
        // 初始化
        InitFrame.initFromDB();

        // 获取任务和资源信息
        taskList = InitFrame.getTaskList();
        vehicleList = InitFrame.getVehicleList();
        rsu = InitFrame.getRSU();
        cloud = InitFrame.getCloud();

        taskSizeFromDB = taskList.size();
        vehicleSizeFromDB = vehicleList.size();
        log.warn("车辆数目: " + vehicleSizeFromDB + "\n任务数量: " + taskSizeFromDB);

        int numParticles = 50; // 粒子数量
        int numDimensions = taskSizeFromDB; // 问题的维度，根据实际问题设置
        int numIterations = 100; // 迭代次数
        double inertiaWeight = 0.5; // 惯性权重
        double cognitiveWeight = 1.5; // 个体认知权重
        double socialWeight = 1.5; // 群体社会权重
        int bound1 = vehicleSizeFromDB;
        double[][] bounds2 = new double[numDimensions][2];
        int[] freqRemainArr = new int[vehicleSizeFromDB + 2];
        freqRemainArr[0] = (int) cloud.getFreqRemain();
        freqRemainArr[1] = (int) rsu.getFreqRemain();
        for (int i = 0; i < vehicleSizeFromDB; i++) {
            freqRemainArr[i + 2] = (int) vehicleList.get(i).getFreqRemain();
        }
        for (int i = 0; i < taskSizeFromDB; i++) {
            unloadArr.add(taskList.get(i).getVehicleID());
        }
        Random random = new Random();
        // PSO资源分配数组的边界设定
        for (int i = 0; i < numDimensions; i++) {
            bounds2[i][0] = 100;
            // bounds2[i][1] = 600;
            // bounds2[i][1] = random.nextInt(500) + 100;
            bounds2[i][1] = Math.min(Constants.MAX_POS_PARTICLE,
                    freqRemainArr[unloadArr.get(i) + 1]);
            // bounds2[i][1] = freqRemainArr[unloadArr.get(i)];
        }


        List<Double> uss_avg_list = new ArrayList<>();
        List<Double> energy_avg_list = new ArrayList<>();

        int uss_list_size = 0;
        while (uss_list_size < 200) {

            PSOCO psoco = new PSOCO(numParticles, numDimensions,
                    bound1 - 1, bounds2,
                    inertiaWeight, cognitiveWeight, socialWeight);

            psoco.optimize(numIterations);
            int pareto_size = psoco.getParetoFront().size();

            for (int i = 0; i < pareto_size; i++) {
                uss_avg_list.add(psoco.getParetoFront().get(i)[0]);
                energy_avg_list.add(-psoco.getParetoFront().get(i)[1]);
            }

            uss_list_size = uss_avg_list.size();
        }


        log.warn("uss:" + uss_avg_list.toString());
        log.warn("energy:" + energy_avg_list.toString());

        /*
        int[] unload_arr = new int[numDimensions];
        int[] freq_arr = new int[numDimensions];
        double[] position = psoco.getParetoFrontPos().get(psoco.getParetoFrontPos().size() - 1);
        for (int i = 0; i < numDimensions; i++) {
            unload_arr[i] = (int) position[i];
            freq_arr[i] = (int) position[i + numDimensions];
        }
        log.info("psoco.getParetoFrontPos().size(): " + psoco.getParetoFrontPos().size());
        log.info("unload_arr:\n" + Arrays.toString(unload_arr));
        log.info("freq_arr:\n" + Arrays.toString(freq_arr));

        for (int i = 0; i < numDimensions; i++) {
            freqRemainArr[unload_arr[i] + 1] -= freq_arr[i];
        }

        log.info("Freq Remain Arr :\n " + Arrays.toString(freqRemainArr));

        List<Integer> unloadList = new ArrayList<>();
        List<Integer> freqList = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            unloadList.add(unload_arr[i]);
            freqList.add(freq_arr[i]);
        }
        // 计算任务 --- 时间
        List<Double> taskExecTimeList = new ArrayList<>();
        Formula.calculateTaskExecTime(taskList, unloadList, freqList, taskExecTimeList);
        List<Double> taskUplinkTime2RSUList = new ArrayList<>();
        Formula.calculateTaskTransTime4Uplink2RSU(taskList, unloadList, taskUplinkTime2RSUList);
        List<Double> taskUplinkTime2CloudList = new ArrayList<>();
        Formula.calculateTaskTransTime4UplinkR2C(taskList, unloadList, taskUplinkTime2CloudList);
        List<Double> taskCostTimeList = new ArrayList<>();
        Formula.calculateTaskCostTime(taskList, unloadList, freqList, taskCostTimeList);

        taskExecTimeList = FormatData.getEffectiveValueList4Digit(taskExecTimeList, 3);
        taskUplinkTime2RSUList = FormatData.getEffectiveValueList4Digit(taskUplinkTime2RSUList, 3);
        taskUplinkTime2CloudList = FormatData.getEffectiveValueList4Digit(taskUplinkTime2CloudList, 3);
        taskCostTimeList = FormatData.getEffectiveValueList4Digit(taskCostTimeList, 3);

        log.info("taskExecTimeList: \n" + taskExecTimeList);
        log.info("taskUplinkTime2RSUList: \n" + taskUplinkTime2RSUList);
        log.info("taskUplinkTime2CloudList: \n" + taskUplinkTime2CloudList);
        log.info("taskCostTimeList: \n" + taskCostTimeList);

        // 计算任务 --- 能耗
        List<Double> taskExecEnergyList = new ArrayList<>();
        Formula.calculateTaskExecEnergy(taskList, unloadList, freqList, taskExecEnergyList);
        List<Double> taskUplinkEnergy2RSUList = new ArrayList<>();
        Formula.calculateTaskUplinkEnergy2RSU(taskList, unloadList, freqList,
                taskUplinkTime2RSUList, taskUplinkEnergy2RSUList);
        List<Double> taskUplinkEnergy2CloudList = new ArrayList<>();
        Formula.calculateTaskUplinkEnergy2Cloud(taskList, unloadList, freqList,
                taskUplinkTime2CloudList, taskUplinkEnergy2CloudList);
        // List<Double> taskCostTimeList = new ArrayList<>();
        // Formula.calculateTaskCostTime(taskList, unloadList, freqList, taskCostTimeList);

        taskExecEnergyList = FormatData.getEffectiveValueList4Digit(taskExecEnergyList, 3);
        taskUplinkEnergy2RSUList = FormatData.getEffectiveValueList4Digit(taskUplinkEnergy2RSUList, 3);
        taskUplinkEnergy2CloudList = FormatData.getEffectiveValueList4Digit(taskUplinkEnergy2CloudList, 3);
        // taskCostTimeList = FormatData.getEffectiveValueList4Digit(taskCostTimeList, 3);

        log.info("taskExecEnergyList: \n" + taskExecEnergyList);
        log.info("taskUplinkEnergy2RSUList: \n" + taskUplinkEnergy2RSUList);
        log.info("taskUplinkEnergy2CloudList: \n" + taskUplinkEnergy2CloudList);
        // log.info("taskCostTimeList: \n" + taskCostTimeList);

         */
    }
}
