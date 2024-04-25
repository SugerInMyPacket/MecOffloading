package utils;

import entity.Cloud;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;

import java.util.*;

public class CheckResourceLimit {

    public static boolean checkCurrVehicleFreqAllocIsEnough(List<Vehicle> vehicleList,
                                                            int vehicleID,
                                                            List<Integer> unloadArr,
                                                            List<Integer> freqAllocArr) {
        Long currVehicleFreqRemain = vehicleList.get(vehicleID - 1).getFreqRemain();
        int len = unloadArr.size();

        int unloadTaskFreqSum = 0;
        for (int i = 0; i < len; i++) {
            if (unloadArr.get(i) != vehicleID) continue;

            unloadTaskFreqSum += freqAllocArr.get(i);
        }
        if (unloadTaskFreqSum > currVehicleFreqRemain) return false;
        return true;
    }

    /**
     * TODO：检查目标车辆 l 的计算资源（频率）是否足够
     * 记录不满足约束的 车辆节点编号 & 所分配任务的索引
     *
     * @param vList       :
     * @param taskList    :
     * @param unloadArr    : λ 卸载决策变量
     * @param freqAllocVar : f 资源分配变量
     *                    lamdaVar的长度 == 所有车辆的任务数量之和
     * @return
     */
    public static void checkVehicleLimit(
            List<Vehicle> vList,
            List<Task> taskList,
            List<Integer> unloadArr,
            List<Integer> freqAllocVar,
            boolean[] vehicleAllocResIsLegal,
            Map<Integer, List<Integer>> vehicleResNotEnoughMap
    ) {

        vehicleResNotEnoughMap.clear();

        int vehicleSize = vList.size();
        int taskSize = taskList.size();

        // 卸载到对应车辆的任务 [vehicleID, [taskID, task-res]]
        // Map<Integer, HashMap<Integer, Integer>> vehicleGetTasksMap = new HashMap<>();
        // 卸载到对应车辆的任务 [vehicleID, taskID]
        Map<Integer, Integer> vehicleUnloadTasksMap = new HashMap<>();

        // key --> vehicleNum ; value : taskListOfVehicle
        // Map<Integer, List<Integer>> vehicleResNotEnoughMap = new HashMap<>();

        for (int i = 0; i < taskSize; i++) {
            // unloadArr 的前 numTasks个索引 表示卸载决策，后numTasks个索引表示分配资源量
            int taskID = i;
            // 车辆的编号是 1 到 n
            if (unloadArr.get(i) > 0) {
                // i ==> unloadArr 索引
                if (!vehicleResNotEnoughMap.containsKey(unloadArr.get(i))) {
                    vehicleResNotEnoughMap.put(unloadArr.get(i), new ArrayList<>());
                }
                // 将分配给编号为 unloadArr.get(i) 的任务添加到 list，记录任务索引编号
                vehicleResNotEnoughMap.get(unloadArr.get(i)).add(taskID);
            }
        }

        // 记录每个车辆  分配的计算量 , vehicle 编号是 1 ~ vehicleSize
        double[] vehicleAllocRes = new double[vehicleSize + 1];
        // 记录每个车辆  分配的计算量 是否合法
        // boolean[] vehicleAllocResIsLegal = new boolean[vehicleSize + 1];
        Arrays.fill(vehicleAllocResIsLegal, true);

        // 遍历任务
        for (int i = 0; i < taskSize; i++) {
            int vehicleNumber = unloadArr.get(i);
            if (vehicleNumber <= 0) continue;

            // if (!vehicleGetTasksMap.containsKey(vehicleNumber)) continue;
            // if(!vehicleUnloadTasksMap.containsKey(vehicleNumber)) continue;

            int taskID = i;
            // if (!vehicleGetTasksMap.get(vehicleNumber).containsKey(taskNumber)) continue;
            // 编号为 vehicleNumber 车辆 的计算量
            // vehicleAllocRes[vehicleNumber] += freqAllocVar.get(vehicleUnloadTasksMap.get(vehicleNumber));
            vehicleAllocRes[vehicleNumber] += freqAllocVar.get(taskID);

            // 判断
            if (vehicleAllocRes[vehicleNumber] > vList.get(vehicleNumber - 1).getFreqRemain()) {
                vehicleAllocResIsLegal[vehicleNumber] = false;
            }
        }

        // TODO：应该将 vehicleAllocResIsLegal[] 返回，以便于实现贪婪修正
        // 不是仅仅返回一个 false
        // return false;
    }

    public static void checkVehicleLimit2(
            List<Vehicle> vList,
            List<Task> taskList,
            List<Integer> unloadArr,
            List<Integer> freqAllocVar,
            boolean[] vehicleAllocResIsLegal,
            Map<Integer, List<Integer>> vehicleResNotEnoughMap
    ) {

        vehicleResNotEnoughMap.clear();

        int vehicleSize = vList.size();
        int taskSize = taskList.size();

        // 卸载到对应车辆的任务 [vehicleID, [taskID, task-res]]
        // Map<Integer, HashMap<Integer, Integer>> vehicleGetTasksMap = new HashMap<>();
        // 卸载到对应车辆的任务 [vehicleID, taskID]
        Map<Integer, Integer> vehicleUnloadTasksMap = new HashMap<>();

        // key --> vehicleNum ; value : taskListOfVehicle
        // Map<Integer, List<Integer>> vehicleResNotEnoughMap = new HashMap<>();

        for (int i = 0; i < taskSize; i++) {
            // unloadArr 的前 numTasks个索引 表示卸载决策，后numTasks个索引表示分配资源量
            // 车辆的编号是 1 到 n
            if (unloadArr.get(i) > 0) {
                // i ==> unloadArr 索引

                /**
                 * note: 修改
                 * 1、编号为 unloadArr.get(i) 的车还没有对应的map，new
                 * 2、已有map，直接加入对应任务
                 */
                // vehicleGetTasksMap.put(unloadArr.get(i), new HashMap<>(i, freqAllocVar.get(i)));
                vehicleUnloadTasksMap.put(unloadArr.get(i), i);

                /*
                if(!vehicleGetTasksMap.containsKey(unloadArr.get(i))) {
                    vehicleGetTasksMap.put(unloadArr.get(i), new HashMap<>(i, freqAllocVar.get(i)));
                } else {
                    vehicleGetTasksMap.put(unloadArr.get(i), new HashMap<>(i, freqAllocVar.get(i)));
                }
                 */
                if (!vehicleResNotEnoughMap.containsKey(unloadArr.get(i))) {
                    vehicleResNotEnoughMap.put(unloadArr.get(i), new ArrayList<>());
                }
                // 将分配给编号为 unloadArr.get(i) 的任务添加到 list，记录任务索引编号
                vehicleResNotEnoughMap.get(unloadArr.get(i)).add(i);
            }
        }

        // 记录每个车辆  分配的计算量 , vehicle 编号是 1 ~ vehicleSize
        double[] vehicleAllocRes = new double[vehicleSize + 1];
        // 记录每个车辆  分配的计算量 是否合法
        // boolean[] vehicleAllocResIsLegal = new boolean[vehicleSize + 1];
        Arrays.fill(vehicleAllocResIsLegal, true);

        // 遍历任务
        for (int i = 0; i < taskSize; i++) {
            int vehicleNumber = unloadArr.get(i);
            if (vehicleNumber <= 0) continue;

            // if (!vehicleGetTasksMap.containsKey(vehicleNumber)) continue;
            if(!vehicleUnloadTasksMap.containsKey(vehicleNumber)) continue;

            int taskNumber = i;
            // if (!vehicleGetTasksMap.get(vehicleNumber).containsKey(taskNumber)) continue;
            // 编号为 vehicleNumber 车辆 的计算量
            // vehicleAllocRes[vehicleNumber] += vehicleGetTasksMap.get(vehicleNumber).get(taskNumber);
            // vehicleAllocRes[vehicleNumber] += freqAllocVar.get(vehicleUnloadTasksMap.get(vehicleNumber));
            vehicleAllocRes[vehicleNumber] += freqAllocVar.get(taskNumber);

            // 判断
            if (vehicleAllocRes[vehicleNumber] > vList.get(vehicleNumber).getFreqRemain()) {
                vehicleAllocResIsLegal[vehicleNumber] = false;
            }
        }

        // TODO：应该将 vehicleAllocResIsLegal[] 返回，以便于实现贪婪修正
        // 不是仅仅返回一个 false

        // return false;
    }

    /**
     * 检查 rsu 计算资源是否满足分配
     *
     * @param rsu
     * @param taskList
     * @param lamdaVar
     * @param freqAllocVar
     * @return
     */
    public static boolean checkRSULimit(RoadsideUnit rsu, List<Task> taskList,
                                        List<Integer> lamdaVar, List<Integer> freqAllocVar) {
        int len = lamdaVar.size();
        // note:应该取的是任务总数
        int numTasks = taskList.size();
        int sumRes4RSU = 0;
        for (int i = 0; i < numTasks; i++) {
            if (lamdaVar.get(i) == 0) {
                // sumRes4Cloud += lamdaVar.get(2 * i - 1);
                sumRes4RSU += freqAllocVar.get(i);
            }
        }
        if (sumRes4RSU <= rsu.getFreqRemain()) return true;
        return false;
    }

    /**
     * 检查 cloud 资源
     *
     * @param c
     * @param taskList
     * @param lamdaVar
     * @param freqAllocVar
     * @return
     */
    public static boolean checkRSULimit(Cloud c, List<Task> taskList,
                                        List<Integer> lamdaVar, List<Integer> freqAllocVar) {
        int len = lamdaVar.size();
        // note:应该取的是任务总数
        int numTasks = taskList.size();
        int sumRes4Cloud = 0;
        for (int i = 0; i < numTasks; i++) {
            if (lamdaVar.get(i) == -1) {
                // sumRes4Cloud += lamdaVar.get(2 * i - 1);
                sumRes4Cloud += freqAllocVar.get(i);
            }
        }
        if (sumRes4Cloud <= c.getFreqRemain()) return true;
        return false;
    }
}
