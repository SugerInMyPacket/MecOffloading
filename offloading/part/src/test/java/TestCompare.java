import compare.psoco.PSOCO;
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
import utils.NumUtil;

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
    // 卸载决策
    static List<Double> unloadRatioArr = new ArrayList<>();
    // 资源分配
    static List<Integer> freqAllocArr = new ArrayList<>();
    static List<Integer> freqAllocArrLocal = new ArrayList<>();
    static List<Integer> freqAllocArrRemote = new ArrayList<>();

    /**
    * @Data 2024-04-21
    */
    @Test
    public void testPSOCOA_Part() {

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
        double[][] bounds3 = new double[numDimensions][2];  // Local Freq 边界
        double[][] bounds4 = new double[numDimensions][2];  // Remote Freq 边界
        int[] freqRemainArr = new int[vehicleSizeFromDB + 2];
        freqRemainArr[0] = (int) cloud.getFreqRemain();
        freqRemainArr[1] = (int) rsu.getFreqRemain();
        for (int i = 0; i < vehicleSizeFromDB; i++) {
            freqRemainArr[i + 2] = (int) vehicleList.get(i).getFreqRemain();
        }


        // PSO资源分配数组的边界设定
        for (int i = 0; i < numDimensions; i++) {
            bounds3[i][0] = 100;
            bounds4[i][0] = 100;
            // bounds3[i][1] = Math.min(Constants.MAX_POS_PARTICLE, freqRemainArr[unloadArr.get(i) + 1]);
            // bounds4[i][1] = Math.min(Constants.MAX_POS_PARTICLE, freqRemainArr[unloadArr.get(i) + 1]);
            bounds3[i][1] = 500;
            bounds4[i][1] = 500;
        }

        List<Double> uss_avg_list = new ArrayList<>();
        List<Double> energy_avg_list = new ArrayList<>();

        boolean flag_stop = false;
        while (!flag_stop) {

            Random random = new Random();
            for (int i = 0; i < taskSizeFromDB; i++) {
                // unloadArr.add(taskList.get(i).getVehicleID());
                unloadArr.add(random.nextInt(vehicleSizeFromDB));
            }

            PSOCO psoco = new PSOCO(numParticles, numDimensions,
                    bound1 - 1, bounds3, bounds4,
                    inertiaWeight, cognitiveWeight, socialWeight);

            psoco.optimize(numIterations);

            int pareto_size = psoco.getParetoFront().size();
            for (int i = 0; i < pareto_size; i++) {
                uss_avg_list.add(psoco.getParetoFront().get(i)[0]);
                energy_avg_list.add(-psoco.getParetoFront().get(i)[1]);
            }

            if (uss_avg_list.size() >= 200) {
                flag_stop = true;
            }

        }

        log.warn("uss_avg_list: \n" + uss_avg_list.toString());
        log.warn("energy_avg_list: \n" + energy_avg_list.toString());

        // log.info("unload arr: \n" + Arrays.toString(psoco.getParetoFrontPos().get(0)));
    }

    /**
    * @Data 2024-04-21
    */
    @Test
    public void testGreedy_Part() {
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
                freqAllocArrLocal.clear();
                freqAllocArrRemote.clear();
                for (int k = 0; k < taskSizeFromDB; k++) {
                    double rk = random.nextDouble();
                    if (rk < 0.6) {
                        unloadArr.add(random.nextInt(vehicleSizeFromDB + 1) - 1);
                    } else {
                        unloadArr.add(taskList.get(j).getVehicleID());
                    }
                    // unloadArr.add(taskList.get(j).getVehicleID());
                    unloadRatioArr.add(NumUtil.random(0.1, 0.9));
                    freqAllocArrLocal.add(random.nextInt((j % 8 + 2) * 50) + 100);
                    freqAllocArrRemote.add(random.nextInt((j % 8 + 2) * 50) + 100);
                }

                List<Double> taskUssList = new ArrayList<>();
                List<Double> taskEnergyList = new ArrayList<>();

                // 贪婪修正
                // RevisePolicy.reviseUnloadArr(taskList, vehicleList, rsu, unloadArr, freqAllocArr);
                RevisePolicy.reviseUnloadArrRemote(taskList, vehicleList, rsu, unloadArr, freqAllocArrRemote);

                taskUssList = Formula.getUSS4TaskList(taskList, unloadArr, unloadRatioArr, freqAllocArrLocal, freqAllocArrRemote);
                taskEnergyList = Formula.getEnergy4TaskList(taskList, unloadArr, unloadRatioArr, freqAllocArrLocal, freqAllocArrRemote);

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
    * @Data 2024-04-22
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

        List<Double> ussAvgList = new ArrayList<>();
        List<Double> energyAvgList = new ArrayList<>();
        boolean flag_stop = false;
        int count = 0;
        while (!flag_stop) {
            ++count;
            int round = 200;
            double[] ussAvgArr = new double[round];
            double[] energyAvgArr = new double[round];


            for (int i = 0; i < round; i++) {
                Random random = new Random();
                unloadArr.clear();
                freqAllocArrLocal.clear();
                freqAllocArrRemote.clear();
                for (int j = 0; j < taskSizeFromDB; j++) {
                    int vID = taskList.get(j).getVehicleID();
                    unloadArr.add(vID);
                    // unloadRatioArr.add(NumUtil.random(0.1, 1.0));
                    unloadRatioArr.add(0.0);
                    freqAllocArrLocal.add((int) (random.nextInt(400) * Constants.DATA_SIZE_MULTI_INCREASE));
                    // freqAllocArrLocal.add((int) (random.nextInt(count % 6 + 1) * 100 * Constants.DATA_SIZE_MULTI_INCREASE));
                    freqAllocArrRemote.add((int) (random.nextInt(count % 6 + 1) * 100 * Constants.DATA_SIZE_MULTI_INCREASE));
                }

                List<Double> taskUSSList =
                        Formula.getUSS4TaskList(taskList, unloadArr, unloadRatioArr, freqAllocArrLocal, freqAllocArrRemote);
                List<Double> taskEnergyList =
                        Formula.getEnergy4TaskList(taskList, unloadArr, unloadRatioArr, freqAllocArrLocal, freqAllocArrRemote);

                taskUSSList = FormatData.getEffectiveValueList4Digit(taskUSSList, 5);
                taskEnergyList = FormatData.getEffectiveValueList4Digit(taskEnergyList, 5);
                double uss_total = 0;
                double energy_total = 0;

                for (int j = 0; j < taskSizeFromDB; j++) {
                    uss_total += taskUSSList.get(j);
                    energy_total += taskEnergyList.get(j);
                }

                ussAvgArr[i] = uss_total / taskSizeFromDB;
                energyAvgArr[i] = energy_total / taskSizeFromDB / 35;

                // ussAvgList.add(ussAvgArr[i]);
                // energyAvgList.add(energyAvgArr[i]);
            }

            double uss_total_round = 0;
            double energy_total_round = 0;

            for (int i = 0; i < round; i++) {
                uss_total_round += ussAvgArr[i];
                energy_total_round += energyAvgArr[i];
            }

            ussAvgList.add(uss_total_round / round);
            energyAvgList.add(energy_total_round / round);

            if (ussAvgList.size() >= 200) {
                flag_stop = true;
            }
        }

        ussAvgList = FormatData.getEffectiveValueList4Digit(ussAvgList, 3);
        energyAvgList = FormatData.getEffectiveValueList4Digit(energyAvgList, 3);

        log.warn("uss list: \n" + ussAvgList.toString());
        log.warn("energy list: \n" + energyAvgList.toString());

    }
}
