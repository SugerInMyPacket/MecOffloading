package compare;

import config.InitFrame;
import entity.Cloud;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import lombok.extern.slf4j.Slf4j;
import utils.FormatData;
import utils.Formula;

import java.util.*;
import java.util.function.DoubleToIntFunction;

@Slf4j
public class Test {
    static List<Task> taskList = new ArrayList<>();

    static RoadsideUnit rsu = new RoadsideUnit();
    static Cloud cloud = new Cloud();
    static List<Vehicle> vehicleList = new ArrayList<>();

    static int taskSizeFromDB;
    static int vehicleSizeFromDB;

    // 卸载决策数组
    static List<Integer> unloadArr = new ArrayList<>();
    // 资源分配数组
    static List<Integer> freqAllocArr = new ArrayList<>();


    public static void main(String[] args) {
        // testPSOCO();
        testPSOCO_2();

    }

    public static void testPSOCO() {
        // 初始化
        InitFrame.initFromDB();

        // 获取任务和资源信息
        taskList = InitFrame.getTaskList();
        vehicleList = InitFrame.getVehicleList();
        rsu = InitFrame.getRSU();
        cloud = InitFrame.getCloud();

        taskSizeFromDB = taskList.size();
        vehicleSizeFromDB = vehicleList.size();
        System.out.println("车辆数目: " + vehicleSizeFromDB + "\n任务数量: " + taskSizeFromDB);

        log.info(" ===> 测试PSOCO算法........");
        // PSO 相关参数
        int numParticles = 100; // 粒子数量
        int numDimensions = taskSizeFromDB; // 问题的维度，根据实际问题设置
        int numIterations = 50; // 迭代次数
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
        // PSO资源分配数组的边界设定
        for (int i = 0; i < numDimensions; i++) {
            bounds2[i][0] = 100;
            bounds2[i][1] = 600;
            // bounds2[i][1] = freqRemainArr[unloadArr.get(i)];
        }

        PSOCO psoco = new PSOCO(numParticles, numDimensions,
                bound1 - 1, bounds2,
                inertiaWeight, cognitiveWeight, socialWeight);

        psoco.optimize(numIterations);

        // List<double[]> paretoFrontPos = psoco.getParetoFrontPos();
        // System.out.println(Arrays.toString(paretoFrontPos.get(0)));
        int[] unload_arr = new int[numDimensions];
        int[] freq_arr = new int[numDimensions];
        double[] position = psoco.getParetoFrontPos().get(psoco.getParetoFrontPos().size() - 1);
        for (int i = 0; i < numDimensions; i++) {
            unload_arr[i] = (int) position[i];
            freq_arr[i] = (int) position[i + numDimensions];
        }
        log.info("psoco.getParetoFrontPos().size(): " + psoco.getParetoFrontPos().size());
        log.info("unload_arr:" + Arrays.toString(unload_arr));
        log.info("freq_arr:" + Arrays.toString(freq_arr));

        for (int i = 0; i < numDimensions; i++) {
            freqRemainArr[unload_arr[i] + 1] -= freq_arr[i];
        }

        log.info("Freq Remain Arr : " + Arrays.toString(freqRemainArr));

        // 计算任务执行时间
        List<Double> taskExecTimeList = new ArrayList<>();
        Formula.calculateTaskExecTime(taskList, unloadArr, freqAllocArr, taskExecTimeList);
        List<Double> taskUplinkTime2RSUList = new ArrayList<>();
        Formula.calculateTaskTransTime4Uplink2RSU(taskList, unloadArr, taskUplinkTime2RSUList);
        List<Double> taskUplinkTime2CloudList = new ArrayList<>();
        Formula.calculateTaskTransTime4UplinkR2C(taskList, unloadArr, taskUplinkTime2CloudList);

        taskExecTimeList = FormatData.getEffectiveValueList4Digit(taskExecTimeList, 5);
        taskUplinkTime2RSUList = FormatData.getEffectiveValueList4Digit(taskExecTimeList, 5);
        taskUplinkTime2CloudList = FormatData.getEffectiveValueList4Digit(taskExecTimeList, 5);

        log.info("taskExecTimeList: \n" + taskExecTimeList);
        log.info("taskUplinkTime2RSUList: \n" + taskUplinkTime2RSUList);
        log.info("taskUplinkTime2CloudList: \n" + taskUplinkTime2CloudList);
    }

    public static void testPSOCO_2() {
        // 初始化
        InitFrame.initFromDB();

        // 获取任务和资源信息
        taskList = InitFrame.getTaskList();
        vehicleList = InitFrame.getVehicleList();
        rsu = InitFrame.getRSU();
        cloud = InitFrame.getCloud();

        taskSizeFromDB = taskList.size();
        vehicleSizeFromDB = vehicleList.size();

        // PSO 相关参数
        int numParticles = 100; // 粒子数量
        int numDimensions = taskSizeFromDB; // 问题的维度，根据实际问题设置
        int numIterations = 50; // 迭代次数
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
        // PSO资源分配数组的边界设定
        for (int i = 0; i < numDimensions; i++) {
            bounds2[i][0] = 100;
            bounds2[i][1] = 800;
            // bounds2[i][1] = freqRemainArr[unloadArr.get(i)];
        }

        int round = 1;
        double[] ussAvgArr = new double[round];
        double[] energyAvgArr = new double[round];
        for (int r = 0; r < round; r++) {
            PSOCO psoco = new PSOCO(numParticles, numDimensions,
                    bound1 - 1, bounds2,
                    inertiaWeight, cognitiveWeight, socialWeight);

            psoco.optimize(numIterations);

            int paretoFrontValSize = psoco.paretoFront.size();

            // double tempUssTotal = 0;
            // double energyUssTotal = 0;
            // for (int j = 0; j < paretoFrontValSize; j++) {
            //     tempUssTotal += psoco.paretoFront.get(j)[0];
            //     energyUssTotal += psoco.paretoFront.get(j)[1];
            // }

            double tempUss = 0;
            double tempEnergy = 0;
            for (int j = 0; j < paretoFrontValSize; j++) {
                if(tempUss < psoco.paretoFront.get(j)[0]) {
                    tempUss =  psoco.paretoFront.get(j)[0];
                    tempEnergy = psoco.paretoFront.get(j)[1];
                }
            }
            // Random random = new Random();
            // int select = random.nextInt(paretoFrontValSize);
            // tempUss =  psoco.paretoFront.get(select)[0];
            // tempEnergy =  psoco.paretoFront.get(select)[1];
            ussAvgArr[r] = tempUss;
            energyAvgArr[r] = -tempEnergy;
            // energyAvgArr[r] = (-energyUssTotal) /  (double) paretoFrontValSize;

        }

        double uss_total_round = 0;
        double energy_total_round = 0;

        for (int i = 0; i < round; i++) {
            uss_total_round += ussAvgArr[i];
            energy_total_round += energyAvgArr[i];
        }

        log.info("uss_avg: " + FormatData.getEffectiveValue4Digit(uss_total_round / round, 5));
        log.info("energy_avg: " + FormatData.getEffectiveValue4Digit(energy_total_round / round, 5));


    }
}
