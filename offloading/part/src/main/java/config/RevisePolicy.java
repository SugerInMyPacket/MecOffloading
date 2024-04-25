package config;

import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import enums.Constants;
import lombok.extern.slf4j.Slf4j;
import utils.CheckResourceLimit;
import utils.FormatData;
import utils.Formula;
import utils.SortRules;

import java.util.*;

@Slf4j(topic = "REVISE")
public class RevisePolicy {

    // 卸载决策 list
    List<Integer> unloadArr = new ArrayList<>();
    // 资源分配 list
    List<Integer> resArr = new ArrayList<>();



    /**
     * keynote：修改卸载策略 --- Remote
     *
     * @param taskList
     * @param vList
     * @param rsu
     * @param unloadArr
     * @param freqAllocArrRemote
     * @param taskUSSList
     * @param taskEnergyList
     */
    public static void reviseUnloadArrRemote(List<Task> taskList,
                                             List<Vehicle> vList,
                                             RoadsideUnit rsu,
                                             List<Integer> unloadArr,
                                             List<Integer> freqAllocArrRemote,
                                             List<Double> taskUSSList,
                                             List<Double> taskEnergyList
    ) {
        // ========= 修正卸载决策
        // 1、----------------------------- 车辆资源是否足够分配
        boolean[] vehicleAllocResIsLegal = new boolean[Constants.VEHICLE_NUMS + 1];
        // 车辆所分配的任务列表
        Map<Integer, List<Integer>> vehicleResNotEnoughMap = new HashMap<>();


        // 1.1 检查每个资源提供车辆的res是否足够
        CheckResourceLimit.checkAllVehicleFreqLimit4Remote(vList, taskList, unloadArr,
                freqAllocArrRemote, vehicleAllocResIsLegal, vehicleResNotEnoughMap);

        int len = unloadArr.size();

        // 任务gfc值map ==== <taskID, taskGFC>
        Map<Integer, Double> taskGFCMapVehicle = new HashMap<>();

        for (int i = 0; i < len; i++) {
            // 非卸载到车辆的，跳过
            if (unloadArr.get(i) <= 0) continue;

            // 第i个任务卸载的车辆 id
            int currVehicleID = unloadArr.get(i);
            // 资源充足，跳过
            if (vehicleAllocResIsLegal[currVehicleID] == true) continue;

            int taskID = i;
            if (vehicleAllocResIsLegal[currVehicleID] == false) {
                // 1.2 计算资源不足节点所分配到所有任务的gfc值
                calculateCorrectFactor2Vehicle(currVehicleID, taskList, taskGFCMapVehicle,
                        unloadArr, freqAllocArrRemote, taskUSSList, taskEnergyList);

                // 当前车辆需计算的任务id集合
                List<Integer> currVehicleUnloadTaskIDList = vehicleResNotEnoughMap.get(currVehicleID);

                // NOTE：1.3 当前车辆卸载的任务集按照gfc值进行排序
                SortRules.sortListByRules(currVehicleUnloadTaskIDList, taskList);
                // log.info("《currVehicleUnloadTaskIDList》:" + currVehicleUnloadTaskIDList.toString());

                // 卸载到当前车辆的list遍历编号
                int indexUnload2curVehicle = 0;
                // 1.4 当前车辆需计算任务的总freq需求
                double vehicleUnloadTasksFreqNeed = 0.0;
                for (int taskIdOfCurrVehicle : currVehicleUnloadTaskIDList) {
                    vehicleUnloadTasksFreqNeed += freqAllocArrRemote.get(taskIdOfCurrVehicle);
                }
                // 当前车辆剩余的freq
                double currVehicleFreqRemain = vList.get(currVehicleID).getFreqRemain();
                // 1.5 检查当前车辆的freq资源是否满足
                if (!CheckResourceLimit.checkCurrVehicleFreqAllocIsEnoughRemote(vList, taskList, currVehicleID, unloadArr, freqAllocArrRemote)) {
                    while (vehicleUnloadTasksFreqNeed > currVehicleFreqRemain) {
                        // 注意范围
                        if (indexUnload2curVehicle >= currVehicleUnloadTaskIDList.size()) {
                            // sq1：进入此 if 说明将所有卸载到此车辆的任务都修改到了 rsu or cloud
                            // 寻找有无任务足以卸载
                            for (int taskIdOfCurrVehicle : currVehicleUnloadTaskIDList) {
                                if(currVehicleFreqRemain <= 0) break;
                                if (freqAllocArrRemote.get(taskIdOfCurrVehicle) <= currVehicleFreqRemain) {
                                    unloadArr.set(taskIdOfCurrVehicle, currVehicleID);
                                    currVehicleFreqRemain -= freqAllocArrRemote.get(taskIdOfCurrVehicle);
                                }
                            }
                            break;
                        }

                        // 1.6 修改：原本卸载至 当前车辆的任务 ===> RSU
                        unloadArr.set(currVehicleUnloadTaskIDList.get(indexUnload2curVehicle), 0);
                        // 减去此任务的第i个任务所需的freq
                        vehicleUnloadTasksFreqNeed -= freqAllocArrRemote.get(currVehicleUnloadTaskIDList.get(indexUnload2curVehicle));
                        ++indexUnload2curVehicle;
                    }
                }
            }
            // 1.7 修改至资源分配为 true
            vehicleAllocResIsLegal[currVehicleID] = true;
            // 删除
            vehicleResNotEnoughMap.remove(currVehicleID);
        }

        // 2、-------------------------- RSU资源分配是否足够
        if (!CheckResourceLimit.checkRSULimit4Remote(rsu, taskList, unloadArr, freqAllocArrRemote)) {
            // 计算卸载到 rsu 的任务的 gfc
            // List<Double> rsuUnloadTaskGFCList = new ArrayList<>();
            // calculateCorrectFactor2RSU(rsuUnloadTaskGFCList, taskList, unloadArr, freqAllocArr, taskUSSList, taskEnergyList);
            Map<Integer, Double> rsuUnloadTasksGFCMap = new HashMap<>();
            calculateCorrectFactor2RSU(rsuUnloadTasksGFCMap, taskList, unloadArr, freqAllocArrRemote, taskUSSList, taskEnergyList);


            // 记录unload到rsu的任务列表
            List<Integer> rsuUnloadTaskList = new ArrayList<>();
            // 记录分配至rsu的任务所需freq
            Double rsuUnloadTasksFreqNeed = 0.0;

            for (int i = 0; i < len; i++) {
                if (unloadArr.get(i) != 0) continue;
                rsuUnloadTaskList.add(i);
                rsuUnloadTasksFreqNeed += freqAllocArrRemote.get(i);
            }

            SortRules.sortListByRules(rsuUnloadTaskList, taskList);

            int indexUnloadTask2RSU = 0;
            Long rsuRemainFreq = rsu.getFreqRemain();
            while (rsuUnloadTasksFreqNeed > rsuRemainFreq) {
                if (indexUnloadTask2RSU >= rsuUnloadTaskList.size()) {
                    // RSU还存在剩余资源时，判断有无任务足以卸载
                    for (int taskIdOfRSU : rsuUnloadTaskList) {
                        if (rsuRemainFreq <= 0) break;
                        if (freqAllocArrRemote.get(taskIdOfRSU) <= rsuRemainFreq) {
                            unloadArr.set(taskIdOfRSU, 0);
                            rsuRemainFreq -= freqAllocArrRemote.get(taskIdOfRSU);
                        }
                    }
                    break;
                }
                // 修改至 cloud
                unloadArr.set(rsuUnloadTaskList.get(indexUnloadTask2RSU), -1);
                rsuUnloadTasksFreqNeed -= freqAllocArrRemote.get(rsuUnloadTaskList.get(indexUnloadTask2RSU));

                ++indexUnloadTask2RSU;
            }

        }

    }

    public static void reviseUnloadArrRemote(List<Task> taskList,
                                             List<Vehicle> vList,
                                             RoadsideUnit rsu,
                                             List<Integer> unloadArr,
                                             List<Integer> freqAllocArrRemote
    ) {
        // ========= 修正卸载决策
        // 1、----------------------------- 车辆资源是否足够分配
        // 记录freq是否足够分配
        boolean[] vehicleAllocResIsLegal = new boolean[Constants.VEHICLE_NUMS + 1];
        // 车辆所分配的任务列表
        Map<Integer, List<Integer>> vehicleResNotEnoughMap = new HashMap<>();


        // 1.1 检查每个资源提供车辆的res是否足够
        CheckResourceLimit.checkAllVehicleFreqLimit4Remote(vList, taskList, unloadArr,
                freqAllocArrRemote, vehicleAllocResIsLegal, vehicleResNotEnoughMap);

        int len = unloadArr.size();

        // 任务gfc值map ==== <taskID, taskGFC>
        // Map<Integer, Double> taskGFCMapVehicle = new HashMap<>();

        for (int i = 0; i < len; i++) {
            // 非卸载到车辆的，跳过
            if (unloadArr.get(i) <= 0) continue;

            // 第i个任务卸载的车辆 id
            int currVehicleID = unloadArr.get(i);
            int task4VehicleID = taskList.get(i).getVehicleID();

            // TODO: 如果是Local (currVehicleID == task4VehicleID), 如何处理 ?
            if (currVehicleID == task4VehicleID) {
                // ?
                continue;
            }

            // 资源充足，跳过
            if (vehicleAllocResIsLegal[currVehicleID] == true) continue;

            int taskID = i;
            if (vehicleAllocResIsLegal[currVehicleID] == false) {
                // 1.2 计算资源不足节点所分配到所有任务的gfc值
                // calculateCorrectFactor2Vehicle(currVehicleID, taskList, taskGFCMapVehicle,
                //         unloadArr, freqAllocArrRemote, taskUSSList, taskEnergyList);

                // 当前车辆需计算的任务id集合
                List<Integer> currVehicleUnloadTaskIDList = vehicleResNotEnoughMap.get(currVehicleID);

                // NOTE：1.3 当前车辆卸载的任务集按照gfc值进行排序
                SortRules.sortListByRules(currVehicleUnloadTaskIDList, taskList);
                // log.info("《currVehicleUnloadTaskIDList》:" + currVehicleUnloadTaskIDList.toString());

                // 卸载到当前车辆的list遍历编号
                int indexUnload2curVehicle = 0;
                // 1.4 当前车辆需计算任务的总freq需求
                double vehicleUnloadTasksFreqNeed = 0.0;
                for (int taskIdOfCurrVehicle : currVehicleUnloadTaskIDList) {
                    vehicleUnloadTasksFreqNeed += freqAllocArrRemote.get(taskIdOfCurrVehicle);
                }
                // 当前车辆剩余的freq
                double currVehicleFreqRemain = vList.get(currVehicleID - 1).getFreqRemain();
                // 1.5 检查当前车辆的freq资源是否满足
                if (!CheckResourceLimit.checkCurrVehicleFreqAllocIsEnoughRemote(vList, taskList, currVehicleID, unloadArr, freqAllocArrRemote)) {
                    while (vehicleUnloadTasksFreqNeed > currVehicleFreqRemain) {
                        // 注意范围
                        if (indexUnload2curVehicle >= currVehicleUnloadTaskIDList.size()) {
                            // sq1：进入此 if 说明将所有卸载到此车辆的任务都修改到了 rsu or cloud
                            // 寻找有无任务足以卸载
                            for (int taskIdOfCurrVehicle : currVehicleUnloadTaskIDList) {
                                if(currVehicleFreqRemain <= 0) break;
                                if (freqAllocArrRemote.get(taskIdOfCurrVehicle) <= currVehicleFreqRemain) {
                                    unloadArr.set(taskIdOfCurrVehicle, currVehicleID);
                                    currVehicleFreqRemain -= freqAllocArrRemote.get(taskIdOfCurrVehicle);
                                }
                            }
                            break;
                        }


                        // 1.6 修改：原本卸载至 当前车辆的任务 ===> RSU
                        unloadArr.set(currVehicleUnloadTaskIDList.get(indexUnload2curVehicle), 0);
                        // 减去此任务的第i个任务所需的freq
                        vehicleUnloadTasksFreqNeed -= freqAllocArrRemote.get(currVehicleUnloadTaskIDList.get(indexUnload2curVehicle));
                        ++indexUnload2curVehicle;
                    }
                }
            }
            // 1.7 修改至资源分配为 true
            vehicleAllocResIsLegal[currVehicleID] = true;
            // 删除
            vehicleResNotEnoughMap.remove(currVehicleID);
        }

        // 2、-------------------------- RSU资源分配是否足够
        if (!CheckResourceLimit.checkRSULimit4Remote(rsu, taskList, unloadArr, freqAllocArrRemote)) {
            // 计算卸载到 rsu 的任务的 gfc
            // List<Double> rsuUnloadTaskGFCList = new ArrayList<>();
            // calculateCorrectFactor2RSU(rsuUnloadTaskGFCList, taskList, unloadArr, freqAllocArr, taskUSSList, taskEnergyList);

            // Map<Integer, Double> rsuUnloadTasksGFCMap = new HashMap<>();
            // calculateCorrectFactor2RSU(rsuUnloadTasksGFCMap, taskList, unloadArr,
            //         freqAllocArrRemote, taskUSSList, taskEnergyList);


            // 记录unload到rsu的任务列表
            List<Integer> rsuUnloadTaskList = new ArrayList<>();
            // 记录分配至rsu的任务所需freq
            Double rsuUnloadTasksFreqNeed = 0.0;

            for (int i = 0; i < len; i++) {
                if (unloadArr.get(i) != 0) continue;
                rsuUnloadTaskList.add(i);
                rsuUnloadTasksFreqNeed += freqAllocArrRemote.get(i);
            }

            SortRules.sortListByRules(rsuUnloadTaskList, taskList);

            int indexUnloadTask2RSU = 0;
            Long rsuRemainFreq = rsu.getFreqRemain();
            while (rsuUnloadTasksFreqNeed > rsuRemainFreq) {
                if (indexUnloadTask2RSU >= rsuUnloadTaskList.size()) {
                    // RSU还存在剩余资源时，判断有无任务足以卸载
                    for (int taskIdOfRSU : rsuUnloadTaskList) {
                        if (rsuRemainFreq <= 0) break;
                        if (freqAllocArrRemote.get(taskIdOfRSU) <= rsuRemainFreq) {
                            unloadArr.set(taskIdOfRSU, 0);
                            rsuRemainFreq -= freqAllocArrRemote.get(taskIdOfRSU);
                        }
                    }
                    break;
                }
                // 修改至 cloud
                unloadArr.set(rsuUnloadTaskList.get(indexUnloadTask2RSU), -1);
                rsuUnloadTasksFreqNeed -= freqAllocArrRemote.get(rsuUnloadTaskList.get(indexUnloadTask2RSU));

                ++indexUnloadTask2RSU;
            }

        }

    }

    /**
     * 计算RSU所分配到任务的修正因子
     * 𝐺𝐹𝐶=𝑝_𝑚𝑎𝑥/(𝑝_𝑖^𝑗 )∗(𝛼∗(〖𝑢𝑠𝑠〗_(𝑖,𝑗)^𝑙)/𝜃+(1−𝛼)∗(1−(𝐸_(𝑖,𝑗)^𝑙−𝐸_𝑚𝑖𝑛)/(𝐸_𝑚𝑎𝑥−𝐸_𝑚𝑖𝑛 )))
     */
    public static void calculateCorrectFactor2RSU(List<Double> gfcList,
                                                  List<Task> taskList,
                                                  List<Integer> unloadArr,
                                                  List<Integer> freqAllocArr,
                                                  List<Double> taskUSSList,
                                                  List<Double> taskEnergyList) {
        // 1、得到 rsu 节点分配的任务集
        List<Integer> unloadTaskID2RSU = new ArrayList<>();
        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            if (unloadArr.get(i) == 0) {
                unloadTaskID2RSU.add(i);
                // 2、得到每个任务的用户满意度
                double currTaskUSS = taskUSSList.get(i);
                // 3、得到每个任务的处理能耗
                double currTaskEnergy = taskEnergyList.get(i);
                // 4、计算每个任务的GFC值
                double alpha = Constants.OBJ_ALPHA;
                Double theta = Double.valueOf(Constants.USS_THRESHOLD);
                // TO-DO: energy 的读取
                double maxEnergy = getMaxValue(taskEnergyList);
                double minEnergy = getMinValue(taskEnergyList);

                int maxPrior = Formula.getTaskMaxPrior(taskList);
                Task currTask = taskList.get(i);
                int currTaskPrior = currTask.getP();
                // 计算当前任务的gfc value
                double currTaskGFC = maxPrior / currTaskPrior * (alpha * currTaskUSS + (1 - alpha) * (currTaskEnergy - minEnergy) / (maxEnergy - minEnergy));

                gfcList.add(currTaskGFC);
            }
        }

    }

    /**
     * 计算RSU所分配到任务的修正因子
     * 𝐺𝐹𝐶=𝑝_𝑚𝑎𝑥/(𝑝_𝑖^𝑗 )∗(𝛼∗(〖𝑢𝑠𝑠〗_(𝑖,𝑗)^𝑙)/𝜃+(1−𝛼)∗(1−(𝐸_(𝑖,𝑗)^𝑙−𝐸_𝑚𝑖𝑛)/(𝐸_𝑚𝑎𝑥−𝐸_𝑚𝑖𝑛 )))
     */
    public static void calculateCorrectFactor2RSU(Map<Integer, Double> gfcMap,
                                                  List<Task> taskList,
                                                  List<Integer> unloadArr,
                                                  List<Integer> freqAllocArr,
                                                  List<Double> taskUSSList,
                                                  List<Double> taskEnergyList) {
        // 1、得到 rsu 节点分配的任务集
        List<Integer> unloadTaskID2RSU = new ArrayList<>();
        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            if (unloadArr.get(i) == 0) {
                unloadTaskID2RSU.add(i);
                // 2、得到每个任务的用户满意度
                double currTaskUSS = taskUSSList.get(i);
                // 3、得到每个任务的处理能耗
                double currTaskEnergy = taskEnergyList.get(i);
                // 4、计算每个任务的GFC值
                double alpha = Constants.OBJ_ALPHA;
                Double theta = Double.valueOf(Constants.USS_THRESHOLD);
                // TO-DO: energy 的读取
                double maxEnergy = getMaxValue(taskEnergyList);
                double minEnergy = getMinValue(taskEnergyList);

                int maxPrior = Formula.getTaskMaxPrior(taskList);
                Task currTask = taskList.get(i);
                int currTaskPrior = currTask.getP();
                // 计算当前任务的gfc value
                double currTaskGFC = maxPrior / currTaskPrior * (alpha * currTaskUSS + (1 - alpha) * (currTaskEnergy - minEnergy) / (maxEnergy - minEnergy));

                gfcMap.put(i, currTaskGFC);
            }
        }

    }

    /**
     * 计算车辆所分配任务的gfc值
     *
     * @param vehicleID
     * @param taskGFCMap   : <TaskID, TaskGFC>
     * @param unloadArr
     * @param freqAllocArr
     * @param taskUSSList
     */
    public static void calculateCorrectFactor2Vehicle(int vehicleID,
                                                      List<Task> taskList,
                                                      Map<Integer, Double> taskGFCMap,
                                                      List<Integer> unloadArr,
                                                      List<Integer> freqAllocArr,
                                                      List<Double> taskUSSList,
                                                      List<Double> taskEnergyList) {

        // taskGFCMap.clear();

        // ？？？
        // if(taskGFCMap.containsKey(unloadArr.get(vehicleID))) return;

        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            // if(taskGFCMap.containsKey(unloadArr.get(i))) return;
            if (unloadArr.get(i) == vehicleID) {
                int currTaskID = i;

                double alpha = Constants.OBJ_ALPHA;
                Double theta = Double.valueOf(Constants.USS_THRESHOLD);
                // TO-DO: energy 的读取
                double maxEnergy = getMaxValue(taskEnergyList);
                double minEnergy = getMinValue(taskEnergyList);

                double currTaskUSS = taskUSSList.get(i);
                double currTaskEnergy = taskEnergyList.get(i);

                int maxPrior = Formula.getTaskMaxPrior(taskList);
                Task currTask = taskList.get(i);
                int currTaskPrior = currTask.getP();
                // 计算当前任务的gfc value
                double currTaskGFC = maxPrior / currTaskPrior * (alpha * currTaskUSS + (1 - alpha) * (currTaskEnergy - minEnergy) / (maxEnergy - minEnergy));

                taskGFCMap.put(currTaskID, currTaskGFC);
                // TODO：按 GFC 值排序后，修改此节点的卸载决策
            }
        }

    }

    public static Double getMaxValue(List<Double> list) {
        if (list.size() == 0 || list == null) return 0.0;

        Double maxValue = list.get(0);

        int len = list.size();
        for (int i = 1; i < len; i++) {
            maxValue = Math.max(maxValue, list.get(i));
        }

        return maxValue;
    }

    public static Double getMinValue(List<Double> list) {
        if (list.size() == 0 || list == null) return 0.0;

        Double minValue = list.get(0);

        int len = list.size();
        for (int i = 1; i < len; i++) {
            minValue = Math.min(minValue, list.get(i));
        }

        return minValue;
    }

}
