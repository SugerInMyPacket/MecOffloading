package compare.psoco;

import config.InitFrame;
import entity.Cloud;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;

@Slf4j(topic = "PSOCO_Example")
public class PSOCO_Example {
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
        testPSOCO();
    }

    public static void testPSOCO() {
        log.info("test part --- PSOCO ");
        // 初始化
        InitFrame.initFromDB();

        // 获取任务和资源信息
        taskList = InitFrame.getTaskList();
        vehicleList = InitFrame.getVehicleList();
        rsu = InitFrame.getRSU();
        cloud = InitFrame.getCloud();

        taskSizeFromDB = taskList.size();
        vehicleSizeFromDB = vehicleList.size();
        System.out.println("taskSizeFromDB: " + taskSizeFromDB);
        System.out.println("vehicleSizeFromDB: " + vehicleSizeFromDB);

        // PSO 相关参数
        int numParticles = 100; // 粒子数量
        int numDimensions = taskSizeFromDB; // 问题的维度，根据实际问题设置
        int numIterations = 50; // 迭代次数
        double inertiaWeight = 0.5; // 惯性权重
        double cognitiveWeight = 1.5; // 个体认知权重
        double socialWeight = 1.5; // 群体社会权重
        int bound1 = vehicleSizeFromDB;
        double[] bounds2 = new double[2];
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
            bounds3[i][1] = 800;
            bounds4[i][0] = 100;
            bounds4[i][1] = 800;
            // bounds3[i][1] = freqRemainArr[unloadArr.get(i)];
        }

        PSOCO psoco = new PSOCO(numParticles, numDimensions,
                bound1 - 1, bounds3, bounds4,
                inertiaWeight, cognitiveWeight, socialWeight);

        psoco.optimize(numIterations);
        log.info("Finish !");
    }

}
