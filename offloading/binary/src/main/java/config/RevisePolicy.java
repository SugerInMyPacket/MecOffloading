package config;

import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import enums.Constants;
import enums.TaskPolicy;
import lombok.extern.slf4j.Slf4j;
import unload_decision.Chromosome;
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

    public static void sortListByRulesToRevise(List<Integer> taskIdsListOfNode,
                                       List<Task> taskList) {

        int revise_select = TaskPolicy.TASK_LIST_SORT_RULE;

        if (revise_select == 1) {
            // 优先级排序（降序）
            SortRules.sortByTaskPrior(taskIdsListOfNode, taskList);
        } else if (revise_select == 2) {
            // 计算量排序（升序）
            SortRules.sortMinComputationFirstSequence(taskIdsListOfNode, taskList);
        } else if (revise_select == 3) {
            // deadline 排序（升序）
            SortRules.sortMinDeadLineFirstSequence(taskIdsListOfNode, taskList);
        }

    }

    // TODO：根据修正因子 ==> 修改资源分配
    // 1、排序执行
    // 2、比例分配
    public static void reviseResourceAllocation(List<Task> taskList,
                                                List<Vehicle> vList,
                                                RoadsideUnit rsu,
                                                List<Integer> unloadArr,
                                                List<Integer> freqAllocArr) {
        // ========= 修正资源分配
        // 1、----------------------------- 车辆资源分配是否足够
        boolean[] vehicleAllocResIsLegal = new boolean[Constants.VEHICLE_NUMS + 1];
        // 车辆所分配的任务列表
        Map<Integer, List<Integer>> vehicleResNotEnoughMap = new HashMap<>();

        // 检查每个资源提供车辆的res是否足够
        CheckResourceLimit.checkVehicleLimit(vList, taskList, unloadArr, freqAllocArr,
                vehicleAllocResIsLegal, vehicleResNotEnoughMap);

        int len = unloadArr.size();

        // ******** Vehicle
        for (int i = 0; i < len; i++) {
            // 非卸载到车辆的，跳过
            if (unloadArr.get(i) <= 0) continue;

            // 第i个任务卸载的车辆 id
            int currVehicleID = unloadArr.get(i);
            // 如果当前车辆节点的资源足够分配，跳过
            if (vehicleAllocResIsLegal[currVehicleID] == true) continue;

            int taskID = i;
            // 当前车辆节点的剩余资源
            int currVehicleFreqRemain = (int) vList.get(currVehicleID).getFreqRemain();

            if (vehicleAllocResIsLegal[currVehicleID] == false) {

                // 当前车辆需计算的任务id集合
                List<Integer> currVehicleUnloadTaskIDList = vehicleResNotEnoughMap.get(currVehicleID);

                // 按规则排序
                // sortListByRules(currVehicleUnloadTaskIDList, taskList);
                // 按优先级比例重分配
                reallocFreqByPriorRatio(taskList, currVehicleFreqRemain,
                        currVehicleUnloadTaskIDList, freqAllocArr);

            }
            // 修改至资源分配为 true
            vehicleAllocResIsLegal[currVehicleID] = true;
            // 删除
            vehicleResNotEnoughMap.remove(currVehicleID);

        }

        // ******** RSU
        if (!CheckResourceLimit.checkRSULimit(rsu, taskList, unloadArr, freqAllocArr)) {

            // 记录unload到rsu的任务列表
            List<Integer> rsuUnloadTaskList = new ArrayList<>();

            int rsuFreqRemain = (int) rsu.getFreqRemain();

            for (int i = 0; i < len; i++) {
                if (unloadArr.get(i) != 0) continue;
                rsuUnloadTaskList.add(i);
            }

            reallocFreqByPriorRatio(taskList, rsuFreqRemain, rsuUnloadTaskList, freqAllocArr);
        }

    }

    // 按prior比例重分配
    public static void reallocFreqByPriorRatio(List<Task> taskList,
                                               int currNodeFreqRemain,
                                               List<Integer> taskIdsListOfCurrNode,
                                               List<Integer> freqAllocArr) {
        // 当前node待卸载任务id - prior
        Map<Integer, Integer> tasksOfCurrNodeMap = new HashMap<>();
        int size = taskIdsListOfCurrNode.size();

        int tasksPriorSum = 0;
        for (int i = 0; i < size; i++) {
            int taskID = taskIdsListOfCurrNode.get(i);
            int taskPrior = taskList.get(taskID).getP();
            tasksOfCurrNodeMap.put(taskID, taskPrior);
            tasksPriorSum += taskPrior;
        }

        for (int i = 0; i < size; i++) {
            int taskID = taskIdsListOfCurrNode.get(i);
            int taskPrior = taskList.get(taskID).getP();
            // 可得到当前节点freq的比例
            double freqGetRatio = (double) taskPrior / tasksPriorSum;
            // 可获取的freq
            int currTaskGetFreq = (int) (freqGetRatio * currNodeFreqRemain);
            // 修改资源分配
            freqAllocArr.set(taskID, currTaskGetFreq);
        }

    }



    /***
     * 修正 unloadArr
     * @param taskList
     * @param vList
     * @param rsu
     * @param unloadArr
     * @param freqAllocArr
     */
    public static void reviseUnloadArr(List<Task> taskList,
                                       List<Vehicle> vList,
                                       RoadsideUnit rsu,
                                       List<Integer> unloadArr,
                                       List<Integer> freqAllocArr) {

        int revise_select = TaskPolicy.TASK_LIST_SORT_RULE;

        List<Double> taskCostTimeList = new ArrayList<>();
        Formula.calculateTaskCostTime(taskList, unloadArr, freqAllocArr, taskCostTimeList);

        // HashMap<Integer, Double> vehicleSortMap = new HashMap<>();
        int vSize = vList.size();
        int[][] vehicleSortArr = new int[vSize][2];  // v_id - freqRemain
        for (int i = 0; i < vSize; i++) {
            vehicleSortArr[i][0] = vList.get(i).getVehicleID();
            vehicleSortArr[i][1] = (int) vList.get(i).getFreqRemain();
        }
        // Arrays.toString(vehicleSortArr);

        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            int v_id = unloadArr.get(i);
            if (v_id <= 0) continue;
            int freq_need = freqAllocArr.get(i);
            vehicleSortArr[v_id - 1][1] -= freq_need;
        }

        // ========= 修正卸载决策
        // 1、----------------------------- 车辆资源分配是否足够
        boolean[] vehicleAllocResIsLegal = new boolean[Constants.VEHICLE_NUMS + 1];
        // 车辆所分配的任务列表
        Map<Integer, List<Integer>> vehicleResNotEnoughMap = new HashMap<>();


        // 检查每个资源提供车辆的res是否足够
        CheckResourceLimit.checkVehicleLimit(vList, taskList, unloadArr, freqAllocArr,
                vehicleAllocResIsLegal, vehicleResNotEnoughMap);


        // vehicle
        for (int i = 0; i < len; i++) {
            // 非卸载到车辆的，跳过
            if (unloadArr.get(i) <= 0) continue;

            // 第i个任务卸载的车辆 id
            int currVehicleID = unloadArr.get(i);
            // 如果当前车辆节点的资源足够分配，跳过
            if (vehicleAllocResIsLegal[currVehicleID] == true) continue;

            int taskID = i;

            if (vehicleAllocResIsLegal[currVehicleID] == false) {

                // 当前车辆需计算的任务id集合
                List<Integer> currVehicleUnloadTaskIDList = vehicleResNotEnoughMap.get(currVehicleID);

                // 任务排序
                // SortRules.sortListByRules(currVehicleUnloadTaskIDList, taskList);
                SortRules.sortListByRules(currVehicleUnloadTaskIDList, taskList,
                        taskCostTimeList, freqAllocArr, vList.get(currVehicleID - 1).getFreqRemain());
                // if (revise_select == 1) {
                //     // 优先级排序（降序）
                //     SortRules.sortByTaskPrior(currVehicleUnloadTaskIDList, taskList);
                // } else if (revise_select == 2) {
                //     // 计算量排序（升序）
                //     SortRules.sortMinComputationFirstSequence(currVehicleUnloadTaskIDList, taskList);
                // } else if (revise_select == 3) {
                //     // deadline 排序（升序）
                //     SortRules.sortMinDeadLineFirstSequence(currVehicleUnloadTaskIDList, taskList);
                // }

                // 卸载到当前车辆的list遍历编号
                int indexUnload2curVehicle = 0;
                // 当前车辆需计算任务的总freq需求
                double vehicleUnloadTasksFreqNeed = 0.0;
                for (int taskIdOfCurrVehicle : currVehicleUnloadTaskIDList) {
                    vehicleUnloadTasksFreqNeed += freqAllocArr.get(taskIdOfCurrVehicle);
                }

                // 当前车辆剩余的freq
                double currVehicleFreqRemain = vList.get(currVehicleID - 1).getFreqRemain();
                // 检查当前车辆的freq资源是否满足
                if (!CheckResourceLimit.checkCurrVehicleFreqAllocIsEnough(vList, currVehicleID, unloadArr, freqAllocArr)) {


                    while (vehicleUnloadTasksFreqNeed > currVehicleFreqRemain) {
                        // 注意范围
                        if (indexUnload2curVehicle >= currVehicleUnloadTaskIDList.size()) {
                            // sq1：进入此 if 说明将所有卸载到此车辆的任务都修改到了 rsu or cloud
                            // TODO：需要进一步合理优化
                            // sq2： 还存在计算资源能够计算一些任务，应该排序后卸载一些任务到当前node
                            for (int taskIdOfCurrVehicle : currVehicleUnloadTaskIDList) {
                                if(currVehicleFreqRemain <= 0) break;
                                if (freqAllocArr.get(taskIdOfCurrVehicle) <= currVehicleFreqRemain) {
                                    unloadArr.set(taskIdOfCurrVehicle, currVehicleID);
                                    currVehicleFreqRemain -= freqAllocArr.get(taskIdOfCurrVehicle);
                                }
                            }

                            break;
                        }

                        // TODO: 资源节点排序
                        // SortRules.sortVehicleByMAXFRF(vList);
                        int vListSize = vList.size();
                        int travel_dis = 0;

                        /**/
                        SortRules.vehicleArrSortByRules(vehicleSortArr, vList, unloadArr, freqAllocArr);

                        for (int j = 0; j < vListSize; j++) {
                            int tempFreqNeed = freqAllocArr.get(currVehicleUnloadTaskIDList.get(indexUnload2curVehicle));
                            if (vehicleSortArr[j][1] >= tempFreqNeed) {
                                // 修改：原本卸载至 当前车辆的任务 ===> 车辆 vehicleSortArr[j][0]
                                unloadArr.set(currVehicleUnloadTaskIDList.get(indexUnload2curVehicle), vehicleSortArr[j][0]);
                                vehicleSortArr[j][1] -= tempFreqNeed;
                            } else {
                                ++travel_dis;
                            }
                        }

                        if (travel_dis >= vListSize - 1) {
                            // 修改：原本卸载至 当前车辆的任务 ===> RSU
                            unloadArr.set(currVehicleUnloadTaskIDList.get(indexUnload2curVehicle), 0);
                        }


                        // 修改：原本卸载至 当前车辆的任务 ===> RSU
                        // unloadArr.set(currVehicleUnloadTaskIDList.get(indexUnload2curVehicle), 0);
                        // 减去此任务的第i个任务所需的 freq
                        vehicleUnloadTasksFreqNeed -= freqAllocArr.get(currVehicleUnloadTaskIDList.get(indexUnload2curVehicle));
                        ++indexUnload2curVehicle;
                    }
                }
            }
            // 修改至资源分配为 true
            vehicleAllocResIsLegal[currVehicleID] = true;
            // 删除
            vehicleResNotEnoughMap.remove(currVehicleID);
        }

        // 2、-------------------------- RSU资源分配是否足够
        if (!CheckResourceLimit.checkRSULimit(rsu, taskList, unloadArr, freqAllocArr)) {

            // 记录unload到rsu的任务列表
            List<Integer> rsuUnloadTaskList = new ArrayList<>();
            // 记录分配至rsu的任务所需freq
            Double rsuUnloadTasksFreqNeed = 0.0;

            for (int i = 0; i < len; i++) {
                if (unloadArr.get(i) != 0) {
                    continue;
                } else {
                    rsuUnloadTaskList.add(i);
                    rsuUnloadTasksFreqNeed += freqAllocArr.get(i);
                }
            }

            // SortRules.sortListByRules(rsuUnloadTaskList, taskList);
            SortRules.sortListByRules(rsuUnloadTaskList, taskList, taskCostTimeList, freqAllocArr, rsu.getFreqRemain());

            // if (revise_select == 1) {
            //     // 优先级排序（降序）
            //     SortRules.sortByTaskPrior(rsuUnloadTaskList, taskList);
            // } else if (revise_select == 2) {
            //     // 计算量排序（升序）
            //     SortRules.sortMinComputationFirstSequence(rsuUnloadTaskList, taskList);
            // } else if (revise_select == 3) {
            //     // deadline 排序（升序）
            //     SortRules.sortMinDeadLineFirstSequence(rsuUnloadTaskList, taskList);
            // }

            int indexUnloadTask2RSU = 0;
            Long rsuRemainFreq = rsu.getFreqRemain();
            while (rsuUnloadTasksFreqNeed > rsuRemainFreq) {
                if (indexUnloadTask2RSU >= rsuUnloadTaskList.size()) {
                    // RSU还存在剩余资源时，判断有无任务足以卸载
                    for (int taskIdOfRSU : rsuUnloadTaskList) {
                        if (rsuRemainFreq <= 0) break;
                        if (freqAllocArr.get(taskIdOfRSU) <= rsuRemainFreq) {
                            unloadArr.set(taskIdOfRSU, 0);
                            rsuRemainFreq -= freqAllocArr.get(taskIdOfRSU);
                        }
                    }
                    break;
                }
                // 修改至 cloud
                unloadArr.set(rsuUnloadTaskList.get(indexUnloadTask2RSU), -1);
                rsuUnloadTasksFreqNeed -= freqAllocArr.get(rsuUnloadTaskList.get(indexUnloadTask2RSU));
                ++indexUnloadTask2RSU;
            }
        }

    }

    // 按 GFC 修正卸载策略
    public static void reviseUnloadArr(List<Task> taskList,
                                       List<Vehicle> vList,
                                       RoadsideUnit rsu,
                                       List<Integer> unloadArr,
                                       List<Integer> freqAllocArr,
                                       List<Double> taskUSSList,
                                       List<Double> taskEnergyList
    ) {
        // ========= 修正卸载决策
        // 1、----------------------------- 车辆资源分配是否足够
        boolean[] vehicleAllocResIsLegal = new boolean[Constants.VEHICLE_NUMS + 1];
        // 车辆所分配的任务列表
        Map<Integer, List<Integer>> vehicleResNotEnoughMap = new HashMap<>();


        // 检查每个资源提供车辆的res是否足够
        CheckResourceLimit.checkVehicleLimit(vList, taskList, unloadArr, freqAllocArr,
                vehicleAllocResIsLegal, vehicleResNotEnoughMap);

        int len = unloadArr.size();

        // 任务gfc值map ==== <taskID, taskGFC>
        Map<Integer, Double> taskGFCMapVehicle = new HashMap<>();

        for (int i = 0; i < len; i++) {
            // 非卸载到车辆的，跳过
            if (unloadArr.get(i) <= 0) continue;

            // 第i个任务卸载的车辆 id
            int currVehicleID = unloadArr.get(i);
            // 如果当前车辆节点的资源足够分配，跳过
            if (vehicleAllocResIsLegal[currVehicleID] == true) continue;

            int taskID = i;
            if (vehicleAllocResIsLegal[currVehicleID] == false) {
                // 计算资源不足节点所分配到所有任务的gfc值
                calculateCorrectFactor2Vehicle(currVehicleID, taskList, taskGFCMapVehicle,
                        unloadArr, freqAllocArr, taskUSSList, taskEnergyList);

                List<Double> tempGfcList = new ArrayList<>();
                for (Map.Entry<Integer, Double> entry : taskGFCMapVehicle.entrySet()) {
                    tempGfcList.add(FormatData.getEffectiveValue4Digit(entry.getValue(), 5));
                }
                // log.info("...... 《tempGfcList》 : " + tempGfcList.toString());

                // 当前车辆需计算的任务id集合
                List<Integer> currVehicleUnloadTaskIDList = vehicleResNotEnoughMap.get(currVehicleID);
                // 按 GFC 值升序
                currVehicleUnloadTaskIDList.sort((o1, o2) -> {
                    double ratio1 = taskGFCMapVehicle.get(o1);
                    double ratio2 = taskGFCMapVehicle.get(o2);
                    if (ratio1 == ratio2) return 0;
                    return ratio1 > ratio2 ? 1 : -1;  // 从小到大
                });

                // log.info("《currVehicleUnloadTaskIDList》:" + currVehicleUnloadTaskIDList.toString());

                // 卸载到当前车辆的list遍历编号
                int indexUnload2curVehicle = 0;
                // 当前车辆需计算任务的总freq需求
                double vehicleUnloadTasksFreqNeed = 0.0;
                for (int taskIdOfCurrVehicle : currVehicleUnloadTaskIDList) {
                    vehicleUnloadTasksFreqNeed += freqAllocArr.get(taskIdOfCurrVehicle);
                }
                // 当前车辆剩余的freq
                double currVehicleFreqRemain = vList.get(currVehicleID - 1).getFreqRemain();
                // 检查当前车辆的freq资源是否满足
                if (!CheckResourceLimit.checkCurrVehicleFreqAllocIsEnough(vList, currVehicleID, unloadArr, freqAllocArr)) {
                    while (vehicleUnloadTasksFreqNeed > currVehicleFreqRemain) {
                        // 注意范围
                        if (indexUnload2curVehicle >= currVehicleUnloadTaskIDList.size()) {
                            // sq1：进入此 if 说明将所有卸载到此车辆的任务都修改到了 rsu or cloud
                            // TODO：需要进一步合理优化

                            break;
                        }
                        // keynote：sq2： 还存在计算资源能够计算一些任务，应该排序后卸载一些任务到当前node

                        // 修改：原本卸载至 当前车辆的任务 ===> RSU
                        unloadArr.set(currVehicleUnloadTaskIDList.get(indexUnload2curVehicle), 0);
                        // 减去此任务的第i个任务所需的freq
                        vehicleUnloadTasksFreqNeed -= freqAllocArr.get(currVehicleUnloadTaskIDList.get(indexUnload2curVehicle));
                        ++indexUnload2curVehicle;
                    }
                }
            }
            // 修改至资源分配为 true
            vehicleAllocResIsLegal[currVehicleID] = true;
            // 删除
            vehicleResNotEnoughMap.remove(currVehicleID);
        }

        /*
        // int size = vehicleResNotEnoughMap.size();
        for (int i = 0; i < len; i++) {
            if(unloadArr.get(i) <= 0) continue;

            int currVehicleID = unloadArr.get(i);
            List<Integer> currVehicleUnloadTaskIDList = vehicleResNotEnoughMap.get(currVehicleID);
            currVehicleUnloadTaskIDList.sort(new Comparator<Integer>() {
                @Override
                public int compare(Integer o1, Integer o2) {
                    double ratio1 = taskGFCMapVehicle.get(o1);
                    double ratio2 = taskGFCMapVehicle.get(o2);
                    if(ratio1 == ratio2) return 0;
                    return ratio1 > ratio2 ? 1 : -1;  // 从小到大
                }
            });

            int indexUnload2currVehcile = 0;
            double vehicleUnloadTasksFreqNeed = 0.0;
            for (int taskID : currVehicleUnloadTaskIDList) {
                vehicleUnloadTasksFreqNeed += freqAllocArr.get(taskID);
            }
            double currVehicleFreqRemain = vList.get(currVehicleID).getFreqRemain();
            if(!CheckResourceLimit.checkCurrVehicleFreqAllocIsEnough(vList, currVehicleID, unloadArr, freqAllocArr)) {
                while (vehicleUnloadTasksFreqNeed > currVehicleFreqRemain) {
                    unloadArr.set(currVehicleUnloadTaskIDList.get(indexUnload2currVehcile), 0);
                    vehicleUnloadTasksFreqNeed -= currVehicleUnloadTaskIDList.get(indexUnload2currVehcile);
                    ++indexUnload2currVehcile;
                }
            }
            // TO-DO: 需要修改，避免多次内循环
            // while (!CheckResourceLimit.checkCurrVehicleFreqAllocIsEnough(vList, currVehicleID, unloadArr, freqAllocArr)) {
            //     // currVehicleID 车辆资源不足分配时，
            //     if (indexUnload2currVehcile >= currVehicleUnloadTaskIDList.size()) break;
            //     // 修正卸载决策到 RSU【0】
            //     unloadArr.set(currVehicleUnloadTaskIDList.get(indexUnload2currVehcile), 0);
            //     ++indexUnload2currVehcile;
            // }
            // 将当前车辆从 vehicleResNotEnoughMap 中移除
            vehicleResNotEnoughMap.remove(currVehicleID);
        }
         */


        // 2、-------------------------- RSU资源分配是否足够
        if (!CheckResourceLimit.checkRSULimit(rsu, taskList, unloadArr, freqAllocArr)) {
            // 计算卸载到 rsu 的任务的 gfc
            // List<Double> rsuUnloadTaskGFCList = new ArrayList<>();
            // calculateCorrectFactor2RSU(rsuUnloadTaskGFCList, taskList, unloadArr, freqAllocArr, taskUSSList, taskEnergyList);
            Map<Integer, Double> rsuUnloadTasksGFCMap = new HashMap<>();
            calculateCorrectFactor2RSU(rsuUnloadTasksGFCMap, taskList, unloadArr, freqAllocArr, taskUSSList, taskEnergyList);


            // 记录unload到rsu的任务列表
            List<Integer> rsuUnloadTaskList = new ArrayList<>();
            // 记录分配至rsu的任务所需freq
            Double rsuUnloadTasksFreqNeed = 0.0;

            for (int i = 0; i < len; i++) {
                if (unloadArr.get(i) != 0) continue;
                rsuUnloadTaskList.add(i);
                rsuUnloadTasksFreqNeed += freqAllocArr.get(i);
            }

            rsuUnloadTaskList.sort(new Comparator<Integer>() {
                @Override
                public int compare(Integer o1, Integer o2) {
                    // double ratio1 = rsuUnloadTaskGFCList.get(o1);
                    // double ratio2 = rsuUnloadTaskGFCList.get(o2);
                    double ratio1 = rsuUnloadTasksGFCMap.get(o1);
                    double ratio2 = rsuUnloadTasksGFCMap.get(o2);
                    if (ratio1 == ratio2) return 0;
                    return ratio1 > ratio2 ? 1 : -1;
                }
            });

            int indexUnloadTask2RSU = 0;
            Long rsuRemainFreq = rsu.getFreqRemain();
            while (rsuUnloadTasksFreqNeed > rsuRemainFreq) {
                if (indexUnloadTask2RSU >= rsuUnloadTaskList.size()) break;
                // 修改至 cloud
                unloadArr.set(rsuUnloadTaskList.get(indexUnloadTask2RSU), -1);
                rsuUnloadTasksFreqNeed -= freqAllocArr.get(rsuUnloadTaskList.get(indexUnloadTask2RSU));
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
