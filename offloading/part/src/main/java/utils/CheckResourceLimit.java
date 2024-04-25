package utils;

import entity.Cloud;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;

import java.util.*;

public class CheckResourceLimit {

    /**
     * 检查编号为 vehicleID 车辆本地任务分配的freq是否足够
     * @param vehicleList
     * @param taskList
     * @param vehicleID
     * @param unloadArr
     *
     * @param freqAllocArrLocal
     * @return
     */
    public static boolean checkCurrVehicleFreqAllocIsEnoughLocal(List<Vehicle> vehicleList,
                                                            List<Task> taskList,
                                                            int vehicleID,
                                                            List<Integer> unloadArr,
                                                            // List<Double> unloadRatioArr,
                                                            List<Integer> freqAllocArrLocal) {

        // 获取编号为 vehicleID 车辆的剩余 freq
        Long currVehicleFreqRemain = vehicleList.get(vehicleID).getFreqRemain();

        int len = taskList.size();

        // 记录卸载到编号为 vehicleID 车辆的 task 所需资源
        int unloadTaskFreqLocalSum = 0;

        for (int i = 0; i < len; i++) {
            Task currTask = taskList.get(i);
            if (currTask.getVehicleID() == vehicleID) {
                unloadTaskFreqLocalSum += freqAllocArrLocal.get(i);
            }
        }

        if (unloadTaskFreqLocalSum > currVehicleFreqRemain) return false;

        return true;
    }

    /**
     * 检查编号为 vehicleID 的车辆接收到的远程任务所分配的freq是否充足
     * @param vehicleList
     * @param taskList
     * @param vehicleID
     * @param unloadArr
     *
     * @param freqAllocArrRemote
     * @return
     */
    public static boolean checkCurrVehicleFreqAllocIsEnoughRemote(List<Vehicle> vehicleList,
                                                                 List<Task> taskList,
                                                                 int vehicleID,
                                                                 List<Integer> unloadArr,
                                                                 // List<Double> unloadRatioArr,
                                                                 List<Integer> freqAllocArrRemote) {

        // 获取编号为 vehicleID 车辆的剩余 freq
        Long currVehicleFreqRemain = vehicleList.get(vehicleID - 1).getFreqRemain();

        int len = taskList.size();

        // 记录卸载到编号为 vehicleID 车辆的 task 所需资源
        int unloadTaskFreqRemoteSum = 0;

        for (int i = 0; i < len; i++) {
            if (unloadArr.get(i) == vehicleID) {
                unloadTaskFreqRemoteSum += freqAllocArrRemote.get(i);
            }
        }

        if (unloadTaskFreqRemoteSum > currVehicleFreqRemain) return false;

        return true;
    }


    /**
     * 判断所有车辆的资源是否满足本地任务的freq分配
     * @param vList
     * @param taskList
     * @param unloadArr
     * @param freqAllocArrLocal : 资源分配变量
     * @param vehicleAllocResIsLegal ： 记录车辆是否满足资源分配
     * @param vehicleResNotEnoughMap ： 记录不满足资源分配的车辆及任务id [vehicleID, taskID-List]
     */
    public static void checkAllVehicleFreqLimit4Local(
            List<Vehicle> vList,
            List<Task> taskList,
            List<Integer> unloadArr,
            List<Integer> freqAllocArrLocal,
            boolean[] vehicleAllocResIsLegal,
            Map<Integer, List<Integer>> vehicleResNotEnoughMap
    ) {

        vehicleResNotEnoughMap.clear();

        int vehicleSize = vList.size();
        int taskSize = taskList.size();


        // 记录每个车辆  分配的计算量 , vehicle 编号是 1 ~ vehicleSize
        double[] vehicleAllocRes = new double[vehicleSize + 1];
        // 记录每个车辆  分配的计算量 是否合法
        Arrays.fill(vehicleAllocResIsLegal, true);

        // 遍历任务
        for (int i = 0; i < taskSize; i++) {
            // 当前任务所属车辆id
            int vehicleIdOfCurrTask = taskList.get(i).getVehicleID();
            // 记录每个车辆的任务id-list
            if (!vehicleResNotEnoughMap.containsKey(vehicleIdOfCurrTask)) {
                vehicleResNotEnoughMap.put(vehicleIdOfCurrTask, new ArrayList<>());
            }
            // 将编号为 vehicleIdOfCurrTask 车辆的所属任务添加到 list，记录任务索引编号
            vehicleResNotEnoughMap.get(vehicleIdOfCurrTask).add(i);

            int taskNumber = i;
            vehicleAllocRes[vehicleIdOfCurrTask] += freqAllocArrLocal.get(i);
            // 判断
            if (vehicleAllocRes[vehicleIdOfCurrTask] > vList.get(vehicleIdOfCurrTask).getFreqRemain()) {
                vehicleAllocResIsLegal[vehicleIdOfCurrTask] = false;
            }
        }


    }

    /**
     * 检查所有车辆接收到的远程任务freq是否足够分配
     * @param vList
     * @param taskList
     * @param unloadArr
     * @param freqAllocArrRemote
     * @param vehicleAllocResIsLegal
     * @param vehicleResNotEnoughMap
     */
    public static void checkAllVehicleFreqLimit4Remote(
            List<Vehicle> vList,
            List<Task> taskList,
            List<Integer> unloadArr,
            List<Integer> freqAllocArrRemote,
            boolean[] vehicleAllocResIsLegal,
            Map<Integer, List<Integer>> vehicleResNotEnoughMap
    ) {

        vehicleResNotEnoughMap.clear();

        int vehicleSize = vList.size();
        int taskSize = taskList.size();


        // 卸载到对应车辆的任务 [vehicleID, taskID]
        // Map<Integer, Integer> vehicleUnloadTasksMap = new HashMap<>();

        // key --> vehicleNum ; value : taskListOfVehicle
        // Map<Integer, List<Integer>> vehicleResNotEnoughMap = new HashMap<>();

        for (int i = 0; i < taskSize; i++) {
            // 车辆的编号是 1 到 n
            if (unloadArr.get(i) > 0) {
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

            int taskID = i;
            // if (!vehicleGetTasksMap.get(vehicleNumber).containsKey(taskNumber)) continue;
            // 编号为 vehicleNumber 车辆 的计算量
            // vehicleAllocRes[vehicleNumber] += vehicleGetTasksMap.get(vehicleNumber).get(taskNumber);
            // vehicleAllocRes[vehicleNumber] += freqAllocArrRemote.get(vehicleUnloadTasksMap.get(vehicleNumber));
            vehicleAllocRes[vehicleNumber] += freqAllocArrRemote.get(taskID);

            // 判断
            if (vehicleAllocRes[vehicleNumber] > vList.get(vehicleNumber - 1).getFreqRemain()) {
                vehicleAllocResIsLegal[vehicleNumber] = false;
            }
        }

    }

    /**
     * 检查编号为 vehicleID 的车辆资源是否满足分配
     * @param vehicleList
     * @param vehicleID
     * @param unloadArr
     * @param freqAllocArr
     * @return
     */
    public static boolean checkCurrVehicleFreqAllocIsEnough(List<Vehicle> vehicleList,
                                                            int vehicleID,
                                                            List<Integer> unloadArr,
                                                            List<Integer> freqAllocArr) {
        // 获取编号为 vehicleID 车辆的剩余 freq
        Long currVehicleFreqRemain = vehicleList.get(vehicleID).getFreqRemain();
        int len = unloadArr.size();

        // 记录卸载到编号为 vehicleID 车辆的 task 所需资源
        int unloadTaskFreqSum = 0;
        for (int i = 0; i < len; i++) {
            if (unloadArr.get(i) != vehicleID) continue;

            unloadTaskFreqSum += freqAllocArr.get(i);
        }
        // 判断
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
     * @param vehicleAllocResIsLegal ： 记录车辆是否满足资源分配
     * @param vehicleResNotEnoughMap ： 记录不满足资源分配的车辆及任务id [vehcileID, taskID-List]
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
        Map<Integer, HashMap<Integer, Integer>> vehicleGetTasksMap = new HashMap<>();

        // 卸载到对应车辆的任务 [vehicleID, taskID]
        Map<Integer, Integer> vehicleUnloadTasksMap = new HashMap<>();

        // key --> vehicleNum ; value : taskListOfVehicle
        // Map<Integer, List<Integer>> vehicleResNotEnoughMap = new HashMap<>();

        for (int i = 0; i < taskSize; i++) {
            // lamdaVar 的前 numTasks个索引 表示卸载决策，后numTasks个索引表示分配资源量
            // 车辆的编号是 1 到 n
            if (unloadArr.get(i) > 0) {
                // i ==> unloadArr 索引

                /**
                 * note: 修改
                 * 1、编号为 lamdaVar.get(i) 的车还没有对应的map，new
                 * 2、已有map，直接加入对应任务
                 */
                vehicleGetTasksMap.put(unloadArr.get(i), new HashMap<>(i, freqAllocVar.get(i)));
                vehicleUnloadTasksMap.put(unloadArr.get(i), i);

                /*
                if(!vehicleGetTasksMap.containsKey(lamdaVar.get(i))) {
                    vehicleGetTasksMap.put(lamdaVar.get(i), new HashMap<>(i, freqAllocVar.get(i)));
                } else {
                    vehicleGetTasksMap.put(lamdaVar.get(i), new HashMap<>(i, freqAllocVar.get(i)));
                }
                 */
                if (!vehicleResNotEnoughMap.containsKey(unloadArr.get(i))) {
                    vehicleResNotEnoughMap.put(unloadArr.get(i), new ArrayList<>());
                }
                // 将分配给编号为 lamdaVar.get(i) 的任务添加到 list，记录任务索引编号
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
            vehicleAllocRes[vehicleNumber] += freqAllocVar.get(vehicleUnloadTasksMap.get(vehicleNumber));
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
     * @param unloadArr
     * @param freqAllocArrRemote : 远程freq分配
     * @return
     */
    public static boolean checkRSULimit4Remote(RoadsideUnit rsu,
                                               List<Task> taskList,
                                               List<Integer> unloadArr,
                                               List<Integer> freqAllocArrRemote) {

        int numTasks = taskList.size();
        // 记录分配到rsu的资源总数
        int sumRes4RSU = 0;
        for (int i = 0; i < numTasks; i++) {
            if (unloadArr.get(i) == 0) {
                sumRes4RSU += freqAllocArrRemote.get(i);
            }
        }
        if (sumRes4RSU <= rsu.getFreqRemain()) return true;
        return false;
    }

    public static boolean checkCloudLimit4Remote(Cloud c,
                                                 List<Task> taskList,
                                                 List<Integer> unloadArr,
                                                 List<Integer> freqAllocArrRemote) {

        int numTasks = taskList.size();
        int sumRes4Cloud = 0;
        for (int i = 0; i < numTasks; i++) {
            if (unloadArr.get(i) == -1) {
                sumRes4Cloud += freqAllocArrRemote.get(i);
            }
        }

        if (sumRes4Cloud <= c.getFreqRemain()) return true;
        return false;
    }

    /**
     * 检查 rsu 计算资源是否满足分配
     *
     * @param rsu
     * @param taskList
     * @param unloadArr
     * @param freqAllocVar
     * @return
     */
    public static boolean checkRSULimit(RoadsideUnit rsu,
                                        List<Task> taskList,
                                        List<Integer> unloadArr,
                                        List<Integer> freqAllocVar) {

        int numTasks = taskList.size();
        // 记录分配到rsu的资源总数
        int sumRes4RSU = 0;
        for (int i = 0; i < numTasks; i++) {
            if (unloadArr.get(i) == 0) {
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
     * @param unloadArr
     * @param freqAllocVar
     * @return
     */
    public static boolean checkCloudLimit(Cloud c, List<Task> taskList,
                                        List<Integer> unloadArr, List<Integer> freqAllocVar) {

        int numTasks = taskList.size();
        int sumRes4Cloud = 0;
        for (int i = 0; i < numTasks; i++) {
            if (unloadArr.get(i) == -1) {
                sumRes4Cloud += freqAllocVar.get(i);
            }
        }
        if (sumRes4Cloud <= c.getFreqRemain()) return true;
        return false;
    }
}
