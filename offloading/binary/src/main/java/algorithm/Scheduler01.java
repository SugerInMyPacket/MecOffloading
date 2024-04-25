package algorithm;

import config.InitFrame;
import entity.Cloud;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import lombok.extern.slf4j.Slf4j;
import resource_allocation.PSO_04;
import unload_decision.GA_03;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

@Slf4j(topic = "Scheduler_01_")
public class Scheduler01 {

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
        // 初始化
        InitFrame.initFromDB();

        // 获取任务和资源信息
        taskList = InitFrame.getTaskList();
        vehicleList = InitFrame.getVehicleList();
        rsu = InitFrame.getRSU();

        taskSizeFromDB = taskList.size();
        vehicleSizeFromDB = vehicleList.size();
        System.out.println("车辆数目: " + vehicleSizeFromDB + "\n任务数量: " + taskSizeFromDB);


        List<Integer> unloadDecisionArrInit = new ArrayList<>(); // 个数应该为 geneLength
        List<Integer> freqAllocArrInit = new ArrayList<>();
        for (int i = 0; i < taskSizeFromDB; i++) {
            // 初始化卸载决策, 随机生成
            unloadDecisionArrInit.add(new Random().nextInt(vehicleSizeFromDB) - 1);
            //TODO: 初始化资源分配, 按聚类所得任务类别进行分配
            freqAllocArrInit.add(new Random().nextInt(200) + 50);
        }


        log.info("TestSchedule ===> 测试 GA algorithm .......");
        // GA 相关参数
        int populationSize = 20; // 种群大小
        double crossoverRate = 0.8; // 交叉概率
        double mutationRate = 0.01; // 变异概率
        int geneLength = taskSizeFromDB; // 染色体长度(维度数)
        int numGenerations = 50; // 迭代代数
        int bound = vehicleSizeFromDB;
        // 执行GA算法 --> unloadArr
        GA_03 ga = new GA_03(populationSize, crossoverRate, mutationRate, geneLength, bound - 1, freqAllocArrInit);
        ga.optimizeChromosomes(numGenerations); // 优化
        // 找到 uss 最高的 pareto 解
        List<int[]> paretoFrontGene_val = ga.getParetoFrontGene();  // pareto解的gene数组
        List<double[]> paretoFront_obj = ga.getParetoFront();  // pareto解的目标值
        int index_unload = 0;
        for (int i = 1; i < paretoFront_obj.size(); i++) {
            if (paretoFront_obj.get(i)[0] > paretoFront_obj.get(i - 1)[0]) {
                index_unload = i;
            }
        }
        // 筛选传输到PSO的卸载决策数组为uss最大的pareto解对应的gene数组
        for (int i = 0; i < taskSizeFromDB; i++) {
            unloadArr.add(paretoFrontGene_val.get(index_unload)[i]);
        }
        log.warn("卸载决策 unloadArr: " + Arrays.toString(unloadArr.toArray()));

        log.info("TestSchedule ===> 测试PSO算法........");
        // PSO 相关参数
        int numParticles = 20; // 粒子数量
        int numDimensions = taskSizeFromDB; // 问题的维度，根据实际问题设置
        int numIterations = 50; // 迭代次数
        double inertiaWeight = 0.5; // 惯性权重
        double cognitiveWeight = 1.5; // 个体认知权重
        double socialWeight = 1.5; // 群体社会权重
        double[][] bounds = new double[numDimensions][2];

        // note：freqRemain ==> bounds[i][1] = freqRemain[unloadArr.get(i) + 1]
        int[] freqRemainArr = new int[vehicleSizeFromDB + 2];
        freqRemainArr[0] = (int) cloud.getFreqRemain();
        freqRemainArr[1] = (int) rsu.getFreqRemain();
        for (int i = 0; i < vehicleSizeFromDB; i++) {
            freqRemainArr[i + 2] = (int) vehicleList.get(i).getFreqRemain();
        }
        // PSO资源分配数组的边界设定
        for (int i = 0; i < numDimensions; i++) {
            bounds[i][0] = 10;
            // bounds[i][1] = 500;
            bounds[i][1] = freqRemainArr[unloadArr.get(i)];
        }
        // PSO 执行
        PSO_04 p = new PSO_04(numParticles, numDimensions, bounds, inertiaWeight, cognitiveWeight,
                socialWeight, unloadArr);
        p.optimize(numIterations);

        List<double[]> paretoFront_val = p.getParetoFront();
        List<double[]> paretoFrontPos = p.getParetoFrontPos();
        int index_resAlloc = 0;
        for (int i = 1; i < paretoFront_val.size(); i++) {
            if (paretoFront_val.get(i)[0] > paretoFront_val.get(i - 1)[0]) {
                index_resAlloc = i;
            }
        }

        for (int i = 0; i < taskSizeFromDB; i++) {
            freqAllocArr.add((int) paretoFrontPos.get(index_resAlloc)[i]);
        }

        System.out.println("最终卸载决策 unloadArr: " + Arrays.toString(unloadArr.toArray()));
        System.out.println("最终资源分配 freqAllocArr: " + Arrays.toString(freqAllocArr.toArray()));
        // log.warn("最终卸载决策 unloadArr: " + Arrays.toString(unloadArr.toArray()));
        // log.warn("最终资源分配 freqAllocArr: " + Arrays.toString(freqAllocArr.toArray()));

    }
}
