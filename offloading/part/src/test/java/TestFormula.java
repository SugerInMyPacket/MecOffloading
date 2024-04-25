import entity.Cloud;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import enums.Constants;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import utils.Formula;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

@Slf4j
public class TestFormula {

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

    // 初始化任务
    public static void initTaskList() {
        for (int i = 0; i < len; i++) {
            Task newTask = new Task();
            newTask.setTaskID(1000 + i);
            newTask.setS(100);
            newTask.setR(0.1f);
            newTask.setC(10);
            newTask.setD(1000);
            newTask.setFactor(5);
            newTask.setI(1);
            newTask.setP(3);
            newTask.setVehicleID(new Random().nextInt(vehicleNums));
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
            unloadArr.add(-1);
            unloadRatioArr.add(new Random().nextDouble());
            // unloadArr.add(new Random().nextInt(vehicleNums + 1) - 1);
            freqAllocArrLocal.add(new Random().nextInt(100) + 50);
            freqAllocArrRemote.add(new Random().nextInt(100) + 50);
        }
    }

    /**
    * @Data 2024-02-07
    */
    @Test
    public void testUss() {
        initResources();
        initTaskList();
        initLamdaVar();

        System.out.println("卸载决策: " + Arrays.toString(unloadArr.toArray()));
        System.out.println("Local 资源分配: " + Arrays.toString(freqAllocArrLocal.toArray()));
        System.out.println("Remote 资源分配: " + Arrays.toString(freqAllocArrRemote.toArray()));

        // 初始化车辆信道增益
        List<Double> gainChannelVehicleList = new ArrayList<>();
        for (int i = 0; i < vehicleNums; i++) {
            gainChannelVehicleList.add(new Random().nextDouble());
        }
        Formula.initGainChannelVehicles(gainChannelVehicleList);


        // 测试任务集满意度计算
        List<Double> taskUSSList = new ArrayList<>();

        taskUSSList = Formula.getUSS4TaskList(taskList, unloadArr, unloadRatioArr, freqAllocArrLocal, freqAllocArrRemote);
        for (int i = 0; i < len; i++) {
            // log.info("++++++" + taskUSSList.get(i) + "++++++");
            System.out.println("++++++" + taskUSSList.get(i) + "++++++");
        }

    }
}
