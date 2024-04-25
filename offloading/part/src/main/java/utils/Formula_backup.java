package utils;

import config.InitFrame;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import enums.Constants;
import lombok.extern.slf4j.Slf4j;

import java.util.*;

@Slf4j
public class Formula_backup {

    // 车辆的信道增益
    // 车辆编号从1开始，gainChannelVehicles[0] 跳过
    static double[] gainChannelVehicles;


    // 找到任务列表中最大优先级
    public static int getTaskMaxPrior(List<Task> taskList) {
        int taskMaxPrior = 0;
        int len = taskList.size();
        for (int i = 0; i < len; i++) {
            taskMaxPrior = Math.max(taskMaxPrior, taskList.get(i).getP());
        }
        return taskMaxPrior;
    }


    /*
     * 初始化车辆的信道增益功率
     * */
    public static void initGainChannelVehicles(List<Double> gainsVehicleList) {
        // int vehicleNums = Constants.VEHICLE_NUMS;
        int vehicleNums = gainsVehicleList.size();
        gainChannelVehicles = new double[vehicleNums + 1];
        gainChannelVehicles[0] = 1.0;

        for (int i = 1; i <= vehicleNums; i++) {
            gainChannelVehicles[i] = gainsVehicleList.get(i - 1);
        }
    }

    // 得到taskList的ussList
    public static List<Double> getUSS4TaskList(List<Task> taskList,
                                               List<Integer> unloadArr,
                                               List<Double> unloadRatioArr,
                                               List<Integer> resAllocArrLocal,
                                               List<Integer> resAllocArrRemote) {
        List<Double> taskUSSList = new ArrayList<>();
        calculateUSS4TaskList(taskList, unloadArr, unloadRatioArr,
                resAllocArrLocal, resAllocArrRemote, taskUSSList);
        return taskUSSList;
    }

    // 计算任务list满意度
    public static void calculateUSS4TaskList(List<Task> taskList,
                                             List<Integer> unloadArr,
                                             List<Double> unloadRatioArr,
                                             List<Integer> resAllocArrLocal,
                                             List<Integer> resAllocArrRemote,
                                             List<Double> taskUSSList) {
        int len = taskList.size();
        taskUSSList.clear();

        // 计算任务的花费时间
        List<Double> taskCostTimeList = new ArrayList<>();
        calculateTaskCostTime(taskList, unloadArr, unloadRatioArr,
                resAllocArrLocal, resAllocArrRemote, taskCostTimeList);
        // 得到task最大优先级
        int taskMaxPrior = getTaskMaxPrior(taskList);

        // 遍历每个任务
        for (int taskIndex = 0; taskIndex < len; taskIndex++) {
            // 当前任务花费时间
            double currTaskCostTime = taskCostTimeList.get(taskIndex);
            // 当前任务
            Task currTask = taskList.get(taskIndex);
            // 属性
            double currTaskDeadline = currTask.getD() / 200.0;
            int currTaskFactor = currTask.getFactor();
            double currTaskMaxDeadline = currTaskFactor * currTaskDeadline;
            int currTaskPrior = currTask.getP();
            // 是否在最大截止期限内处理完成
            if (currTaskCostTime <= currTaskMaxDeadline) {
                double exp
                        = Math.max(0, (currTaskCostTime - currTaskDeadline)) / currTaskMaxDeadline
                        * currTaskPrior / taskMaxPrior;
                // 用户满意度计算阈值 θ
                double theta = Constants.USS_THRESHOLD;
                double currTaskUSS = theta - Math.pow(theta, exp);
                taskUSSList.add(currTaskUSS);
            } else {
                taskUSSList.add(0.0);
            }
        }
    }

    // 得到任务的能耗list
    public static List<Double> getEnergy4TaskList(List<Task> taskList,
                                                  List<Integer> unloadArr,
                                                  List<Double> unloadRatioArr,
                                                  List<Integer> resAllocArrLocal,
                                                  List<Integer> resAllocArrRemote) {
        List<Double> taskEnergyList = new ArrayList<>();
        calculateEnergy4TaskList(taskList, unloadArr, unloadRatioArr,
                resAllocArrLocal, resAllocArrRemote, taskEnergyList);
        return taskEnergyList;
    }

    /**
     * 计算任务的总能耗
     *
     * @param taskList
     * @param taskEnergyList
     */
    public static void calculateEnergy4TaskList(List<Task> taskList,
                                                List<Integer> unloadArr,
                                                List<Double> unloadRatioArr,
                                                List<Integer> resAllocArrLocal,
                                                List<Integer> resAllocArrRemote,
                                                List<Double> taskEnergyList) {
        taskEnergyList.clear();

        // 计算时间【公共部分提取】
        Map<Integer, Double> taskExecTimeMapLocal = new HashMap<>();
        Map<Integer, Double> taskExecTimeMapRemote = new HashMap<>();
        Map<Integer, Double> taskUplinkTimeV2RMap = new HashMap<>();
        Map<Integer, Double> taskUplinkTimeR2CMap = new HashMap<>();
        Map<Integer, Double> taskUplinkTimeV2VMap = new HashMap<>();

        // 计算 time
        // calculateTaskExecTimeLocal(taskList, unloadArr, unloadRatioArr, resAllocArrLocal, taskExecTimeMapLocal);
        // calculateTaskExecTimeRemote(taskList, unloadArr, unloadRatioArr, resAllocArrRemote, taskExecTimeMapRemote);
        // calculateTaskTransTime4Uplink2RSU(taskList, unloadArr, unloadRatioArr, taskUplinkTimeV2RMap);
        // calculateTaskTransTime4UplinkR2C(taskList, unloadArr, unloadRatioArr, taskUplinkTimeR2CMap);
        // calculateTaskTransTime4V2V(taskList, unloadArr, unloadRatioArr, taskUplinkTimeV2VMap);

        // 更新 energy list
        // calculateXXX()
        Map<Integer, Double> taskExecEnergyMapLocal = new HashMap<>();
        Map<Integer, Double> taskExecEnergyMapRemote = new HashMap<>();
        Map<Integer, Double> taskUplinkEnergyV2RMap = new HashMap<>();
        Map<Integer, Double> taskUplinkEnergyR2CMap = new HashMap<>();
        Map<Integer, Double> taskUplinkEnergyV2VMap = new HashMap<>();

        // 计算 energy
        calculateTaskExecEnergyLocal(taskList, unloadArr, unloadRatioArr,
                resAllocArrLocal, taskExecEnergyMapLocal);
        calculateTaskExecEnergyRemote(taskList, unloadArr, unloadRatioArr,
                resAllocArrRemote, taskExecEnergyMapRemote);
        calculateTaskUplinkEnergy2RSU(taskList, unloadArr, unloadRatioArr,
                resAllocArrRemote, taskUplinkTimeV2RMap, taskUplinkEnergyV2RMap);
        calculateTaskUplinkEnergy2Cloud(taskList, unloadArr, unloadRatioArr,
                resAllocArrRemote, taskUplinkTimeR2CMap, taskUplinkEnergyR2CMap);
        calculateTaskTransEnergy4V2V(taskList, unloadArr, unloadRatioArr,
                taskUplinkTimeV2VMap, taskUplinkEnergyV2VMap);

        int len = taskList.size();

        for (int i = 0; i < len; i++) {
            int currTaskID = i;
            double currTaskCostEnergy = 0.0;

            if (taskExecEnergyMapLocal.containsKey(currTaskID)) {
                // Local
                currTaskCostEnergy += taskExecEnergyMapLocal.get(currTaskID);
            }
            if (taskExecEnergyMapRemote.containsKey(currTaskID)) {
                // Remote
                currTaskCostEnergy += taskExecEnergyMapRemote.get(currTaskID);
            }
            if (taskUplinkEnergyV2RMap.containsKey(currTaskID)) {
                currTaskCostEnergy += taskUplinkEnergyV2RMap.get(currTaskID);
            }
            if (taskUplinkEnergyR2CMap.containsKey(currTaskID)) {
                currTaskCostEnergy += taskUplinkEnergyR2CMap.get(currTaskID);
            }
            if (taskUplinkEnergyV2VMap.containsKey(currTaskID)) {
                currTaskCostEnergy += taskUplinkEnergyV2VMap.get(currTaskID);
            }

            taskEnergyList.add(currTaskCostEnergy);
        }

    }

    /**
     * 计算任务总花费时间
     *
     * @param taskList
     * @param unloadArr
     * @param taskCostTimeList
     */
    public static void calculateTaskCostTime(List<Task> taskList,
                                             List<Integer> unloadArr,
                                             List<Double> unloadRatioArr,
                                             List<Integer> resAllocArrLocal,
                                             List<Integer> resAllocArrRemote,
                                             List<Double> taskCostTimeList) {
        // res
        taskCostTimeList.clear();

        // note: 更新 【time list】
        Map<Integer, Double> taskExecTimeMapLocal = new HashMap<>();
        Map<Integer, Double> taskExecTimeMapRemote = new HashMap<>();
        Map<Integer, Double> taskUplinkTimeV2RMap = new HashMap<>();
        Map<Integer, Double> taskUplinkTimeR2CMap = new HashMap<>();
        Map<Integer, Double> taskUplinkTimeV2VMap = new HashMap<>();

        // 计算
        calculateTaskExecTimeLocal(taskList, unloadArr, unloadRatioArr, resAllocArrLocal, taskExecTimeMapLocal);
        calculateTaskExecTimeRemote(taskList, unloadArr, unloadRatioArr, resAllocArrRemote, taskExecTimeMapRemote);
        calculateTaskTransTime4Uplink2RSU(taskList, unloadArr, unloadRatioArr, taskUplinkTimeV2RMap);
        calculateTaskTransTime4UplinkR2C(taskList, unloadArr, unloadRatioArr, taskUplinkTimeR2CMap);
        calculateTaskTransTime4V2V(taskList, unloadArr, unloadRatioArr, taskUplinkTimeV2VMap);

        int len = taskList.size();

        for (int i = 0; i < len; i++) {
            int currTaskID = i;
            Task currTask = taskList.get(i);
            // 任务所述车辆
            int vehicleNum4Task = currTask.getVehicleID();

            double currTaskCostTime = 0.0;
            double currTaskCostTimeLocal = 0.0;
            double currTaskCostTimeRemote = 0.0;

            int unloadObjNode = unloadArr.get(i);
            // note --> 时间： max(Local, Remote)
            if (taskExecTimeMapLocal.containsKey(currTaskID)) {
                // Local
                currTaskCostTimeLocal += taskExecTimeMapLocal.get(currTaskID);
            }
            if (taskExecTimeMapRemote.containsKey(currTaskID)) {
                // Remote
                currTaskCostTimeRemote += taskExecTimeMapRemote.get(currTaskID);
            }
            if (taskUplinkTimeV2RMap.containsKey(currTaskID)) {
                // 卸载到 rsu
                currTaskCostTimeRemote += taskUplinkTimeV2RMap.get(currTaskID);
            }
            if (taskUplinkTimeR2CMap.containsKey(currTaskID)) {
                // 卸载到 cloud
                currTaskCostTimeRemote += taskUplinkTimeR2CMap.get(currTaskID);
            }
            if (taskUplinkTimeV2VMap.containsKey(currTaskID)) {
                // 卸载到 目标车辆
                currTaskCostTimeRemote += taskUplinkTimeV2VMap.get(currTaskID);
            }

            // max (local, remote)
            currTaskCostTime = Math.max(currTaskCostTimeLocal, currTaskCostTimeRemote);

            taskCostTimeList.add(currTaskCostTime);
        }

    }


    /**
     * 计算task本地卸载的执行时间【比例】
     * TODO : 修改 ==> Local 任务处理，freq不足分配时，按【排序策略】处理
     *
     * @param taskList
     * @param unloadRatioArr
     * @param resAllocArrLocal
     * @param taskExecTimeMapLocal
     */
    public static void calculateTaskExecTimeLocal(List<Task> taskList,
                                                  List<Integer> unloadArr,
                                                  List<Double> unloadRatioArr,
                                                  List<Integer> resAllocArrLocal,
                                                  Map<Integer, Double> taskExecTimeMapLocal) {
        // clear 执行时间 list
        taskExecTimeMapLocal.clear();

        int len = taskList.size();

        for (int i = 0; i < len; i++) {
            double unloadRatioLocal = 0.0;

            int currTaskVehicleID = taskList.get(i).getVehicleID();
            int objVehicleID = unloadArr.get(i);
            if (currTaskVehicleID == objVehicleID) {
                // 说明完全卸载到了Local，unloadRatioLocal = unloadRatioArr.get(i);
                unloadRatioLocal =unloadRatioArr.get(i);
            } else {
                // local 卸载的比例
                unloadRatioLocal = 1.0 - unloadRatioArr.get(i);
            }
            // 当前任务
            Task currTask = taskList.get(i);
            int taskSize = currTask.getS();
            float processDensity = currTask.getC();

            // currTask 的 计算需求量
            double processCost = taskSize * processDensity * unloadRatioLocal;
            // 分配给 currTask 的计算资源量
            int resAlloc2currTask = resAllocArrLocal.get(i);

            // 计算执行时间
            double currTaskExecTime = processCost / (double) resAlloc2currTask;

            // 加入 taskExecTimeList
            taskExecTimeMapLocal.put(i, currTaskExecTime);
        }
    }

    /**
     * 当Local freq不足以分配时，排序处理卸载到此节点的任务
     *
     * @param taskList
     * @param unloadArr
     * @param freqAllocArr
     * @param taskExecTimeMap
     */
    public static void calculateTaskExecTimeBySort(List<Task> taskList,
                                                   List<Integer> unloadArr,
                                                   List<Integer> freqAllocArr,
                                                   Map<Integer, Double> taskExecTimeMap) {
        // clear 执行时间 list
        taskExecTimeMap.clear();

        // Map<Integer, Double> tasksExecTimeMap = new HashMap<>();

        int len = taskList.size();

        // 获取卸载到每个节点的任务id-List
        Map<Integer, List<Integer>> nodeUnloadTaskIDsMap = new HashMap<>();
        for (int i = 0; i < len; i++) {
            int unloadNodeID = unloadArr.get(i);
            if (!nodeUnloadTaskIDsMap.containsKey(unloadNodeID)) {
                nodeUnloadTaskIDsMap.put(unloadNodeID, new ArrayList<>());
            }
            nodeUnloadTaskIDsMap.get(unloadNodeID).add(i);
        }

        List<Vehicle> vehicleList = InitFrame.getVehicleList();
        RoadsideUnit rsu = InitFrame.getRSU();

        // 不排序时，任务执行时间
        List<Double> taskExecTimeListNoSort = new ArrayList<>();
        for (int i = 0; i < len; i++) {
            // 当前任务
            Task currTask = taskList.get(i);
            int taskSize = currTask.getS();
            float processDensity = currTask.getC();

            // currTask 的 计算需求量
            double processCost = taskSize * processDensity;
            // 分配给 currTask 的计算资源量
            int resAlloc2currTask = freqAllocArr.get(i);

            // 计算执行时间
            double currTaskExecTime = processCost / (double) resAlloc2currTask;

            // 加入 taskExecTimeList
            taskExecTimeListNoSort.add(currTaskExecTime);
        }

        int nodeSize = vehicleList.size() + 1;
        // 遍历 rsu 和 Vehicle
        for (int nodeID = 0; nodeID < nodeSize; nodeID++) {
            List<Integer> currNodeGetTaskIDsList = new ArrayList<>();
            currNodeGetTaskIDsList = nodeUnloadTaskIDsMap.get(nodeID);

            // sort - currNodeGetTaskIDsList
            SortRules.sortListByRules(currNodeGetTaskIDsList, taskList);

            // 计算排序执行时间
            int currNodeFreqRemain = 0;
            if (nodeID == 0) {
                currNodeFreqRemain = (int) rsu.getFreqRemain();
            } else {
                currNodeFreqRemain = (int) vehicleList.get(nodeID - 1).getFreqRemain();
            }

            // 当前node的任务集所需要的freq
            List<Integer> currNodeTasksNeedFreqList = new ArrayList<>();
            for (int j = 0; j < currNodeGetTaskIDsList.size(); j++) {
                int tempFreqNeed = freqAllocArr.get(currNodeGetTaskIDsList.get(j));
                currNodeTasksNeedFreqList.add(tempFreqNeed);
            }

            // NOTE： freq 不足分配时，需要加上排序靠前任务的 execTime
            int size = currNodeGetTaskIDsList.size();
            // 待处理任务指针
            int task_index_need = 0;
            // 已处理任务指针
            int task_index_already = 0;
            // 当前车辆freq不足以处理时，下一次待处理的task需要加上的已经处理完任务时间
            double needPlusTime = 0.0;

            List<Integer> taskAlreadyExecIdsList = new ArrayList<>();
            Map<Integer, Double> taskAlreadyExecTimeMap = new HashMap();
            while (task_index_need < size) {
                // 按规则排序后，当前要处理的任务id 及 所需 freq
                int tempTaskID = currNodeGetTaskIDsList.get(task_index_need);
                int tempFreqNeed = freqAllocArr.get(tempTaskID);

                if (currNodeFreqRemain >= tempFreqNeed) {
                    // 车辆 freqRemain 减少
                    currNodeFreqRemain -= tempFreqNeed;
                    // tasksExecTimeMap.put(tempTaskID,
                    //         needPlusTime + taskExecTimeListNoSort.get(task_index_need));

                    taskExecTimeMap.put(tempTaskID,
                            needPlusTime + taskExecTimeListNoSort.get(task_index_need));

                    // 当前Node已执行完的任务，及其时间
                    taskAlreadyExecIdsList.add(tempTaskID);
                    taskAlreadyExecTimeMap.put(tempTaskID,
                            needPlusTime + taskExecTimeListNoSort.get(task_index_need));

                    // 待处理任务指针后移
                    ++task_index_need;
                } else {
                    // 当前处理N个任务，NodeFreqRemain不足处理下一个task时
                    if (task_index_already >= task_index_need) {
                        // 说明分配freq超出 当前NodeFreqRemain
                        freqAllocArr.set(tempTaskID, currNodeFreqRemain);
                        continue;
                    }

                    if (taskAlreadyExecIdsList.size() >= 2) {
                        // 已处理的任务id 按处理时间 【升序】排列
                        taskAlreadyExecIdsList.sort(new Comparator<Integer>() {
                            @Override
                            public int compare(Integer id1, Integer id2) {
                                double ratio1 = taskAlreadyExecTimeMap.get(id1);
                                double ratio2 = taskAlreadyExecTimeMap.get(id2);
                                return ratio1 > ratio2 ? 1 : -1;
                            }
                        });
                    }

                    // 找到 already处理任务的最小执行时间对应的任务id
                    int minExecTimeTaskId = taskAlreadyExecIdsList.get(task_index_already);
                    // 当前车辆的freq增加
                    currNodeFreqRemain += freqAllocArr.get(minExecTimeTaskId);

                    // 下一次资源足够时，需要加上的时间
                    needPlusTime = Math.max(needPlusTime,
                            taskAlreadyExecTimeMap.get(minExecTimeTaskId));

                    ++task_index_already;
                }
            }
        }

    }

    /**
     * 计算task远程卸载的执行时间【比例】
     *
     * @param taskList
     * @param unloadRatioArr
     * @param resAllocArrRemote
     * @param taskExecTimeMapRemote
     */
    public static void calculateTaskExecTimeRemote(List<Task> taskList,
                                                   List<Integer> unloadArr,
                                                   List<Double> unloadRatioArr,
                                                   List<Integer> resAllocArrRemote,
                                                   Map<Integer, Double> taskExecTimeMapRemote) {
        // clear 执行时间 list
        taskExecTimeMapRemote.clear();

        int len = taskList.size();

        for (int i = 0; i < len; i++) {
            double unloadRatioRemote = 0.0;

            int currTaskVehicleID = taskList.get(i).getVehicleID();
            int objVehicleID = unloadArr.get(i);
            if (currTaskVehicleID == objVehicleID) {
                // 说明完全卸载到了Local，unloadRatioLocal = unloadRatioArr.get(i);
                unloadRatioRemote = 1.0 - unloadRatioArr.get(i);
            } else {
                // Remote 卸载的比例
                unloadRatioRemote = unloadRatioArr.get(i);
            }

            // 当前任务
            Task currTask = taskList.get(i);
            int taskSize = currTask.getS();
            float processDensity = currTask.getC();

            // currTask 的 计算需求量
            double processCost = taskSize * processDensity * unloadRatioRemote;
            // 分配给 currTask 的计算资源量
            int resAlloc2currTask = resAllocArrRemote.get(i);

            // 计算执行时间
            double currTaskExecTime = processCost / (double) resAlloc2currTask;

            // 加入 taskExecTimeList
            taskExecTimeMapRemote.put(i, currTaskExecTime);
        }
    }


    /**
     * 算需传输到RSU任务的上行传输时间
     *
     * @param taskList
     * @param unloadArr
     * @param unloadRatioArr
     * @param taskUplinkTimeMap2RSU
     */
    public static void calculateTaskTransTime4Uplink2RSU(List<Task> taskList,
                                                         List<Integer> unloadArr,
                                                         List<Double> unloadRatioArr,
                                                         Map<Integer, Double> taskUplinkTimeMap2RSU) {
        taskUplinkTimeMap2RSU.clear();

        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            // 非卸载到 rsu 的任务，跳过
            if (unloadArr.get(i) != 0) continue;

            // Remote 卸载的比例
            double unloadRatioRemote = unloadRatioArr.get(i);
            // 当前任务
            Task currTask = taskList.get(i);
            int taskSize = currTask.getS();

            // 传输速率变量
            // double rate4v2r = getTransRateV2R(currTask.getVehicleID(), unloadArr);
            double rate4v2r = getCurrVehicleRateV2R(currTask.getVehicleID());

            double currTaskUplinkTime = taskSize * unloadRatioRemote / (double) rate4v2r;

            // 记录 currTask 传输时间
            taskUplinkTimeMap2RSU.put(i, currTaskUplinkTime);
        }

    }

    // 计算传输速率
    public static double getCurrVehicleRateV2R(int vehicleID) {
        List<Vehicle> vehicleList = InitFrame.getVehicleList();

        int posX = (int) vehicleList.get(vehicleID - 1).getPosX();

        int currVehicleChannelWidth = Constants.WIDTH_CHANNEL_RSU;

        double rateV2R = currVehicleChannelWidth * Math.log(2 - Math.abs(posX - 200) / 400);

        return rateV2R;

    }

    /**
     * 计算vehicle到rsu的传输速率
     */
    public static double getTransRateV2R(int vehicleID, List<Integer> unloadArr) {

        // 需要传输task到rsu的车辆数目
        int vehicleNums2RSU = 0;
        int len = unloadArr.size();
        for (int i = 0; i < len; i++) {
            if (unloadArr.get(i) <= 0) {
                ++vehicleNums2RSU;
            }
        }

        double widthC2R = Constants.WIDTH_CHANNEL_RSU / (double) vehicleNums2RSU;

        double gainChannelV2R = gainChannelVehicles[vehicleID];
        Long powerTransVehicle = Constants.POWER_TRANS_VEHICLE;

        Long powerNoise = Constants.POWER_NOISE;

        // todo:计算来自其他车辆的无线干扰
        double otherVehiclesNoise = 1.0;

        double transRateV2R
                = widthC2R * Math.log(1 + (powerTransVehicle * gainChannelV2R) / powerNoise
                + otherVehiclesNoise);

        return transRateV2R;
    }

    /**
     * 计算从RSU传输到cloud的上行传输时间
     *
     * @param taskList
     * @param unloadArr
     * @param unloadRatioArr
     * @param taskUplinkTimeMap2Cloud
     */
    public static void calculateTaskTransTime4UplinkR2C(List<Task> taskList,
                                                        List<Integer> unloadArr,
                                                        List<Double> unloadRatioArr,
                                                        Map<Integer, Double> taskUplinkTimeMap2Cloud) {
        taskUplinkTimeMap2Cloud.clear();

        // 传输速率变量
        double rate4r2c = getTransRateR2C();

        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            if (unloadArr.get(i) != -1) {
                continue;
            } else {
                // Remote 卸载的比例
                double unloadRatioRemote = unloadRatioArr.get(i);
                // 当前任务
                Task currTask = taskList.get(i);

                int taskSize = currTask.getS();
                // 计算当前任务的传输时间
                double currTaskUplinkTime = taskSize * unloadRatioRemote / (double) rate4r2c;

                // 记录 currTask 传输时间
                taskUplinkTimeMap2Cloud.put(i, currTaskUplinkTime);
            }
        }

    }

    /**
     * 计算RSU到Cloud的传输速率
     */
    public static double getTransRateR2C() {
        double widthR2C = Constants.WIDTH_CHANNEL_CLOUD;
        Long gainChannelR2C = Constants.GAIN_CHANNEL_R2C;
        Long powerTransRSU = Constants.POWER_TRANS_RSU;
        Long powerNoise = Constants.POWER_NOISE;

        double transRateR2C = widthR2C * Math.log(1 + (powerTransRSU * gainChannelR2C) / powerNoise);

        return transRateR2C;
    }


    /**
     * 计算车辆间任务传输时间
     *
     * @param taskList
     * @param unloadArr
     * @param unloadRatioArr
     * @param taskUplinkTimeMap2Vehicle
     */
    public static void calculateTaskTransTime4V2V(List<Task> taskList,
                                                  List<Integer> unloadArr,
                                                  List<Double> unloadRatioArr,
                                                  Map<Integer, Double> taskUplinkTimeMap2Vehicle) {
        taskUplinkTimeMap2Vehicle.clear();

        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            // 传输到 rsu 的任务，跳过
            // if(unloadArr.get(i) <= 0) continue;
            if (unloadArr.get(i) <= 0) {
                // taskTransTimeListV2V.add(0.0);
                continue;
            }

            // Remote 卸载的比例
            double unloadRatioRemote = unloadRatioArr.get(i);

            // 计算
            Task currTask = taskList.get(i);
            // currTask 所属车辆
            int vehicleOfCurrTask = currTask.getVehicleID();
            // currTask 卸载的目标车辆
            int objOffloadVehicle = unloadArr.get(i);
            // 卸载 local
            if (vehicleOfCurrTask == objOffloadVehicle) {
                // taskTransTimeListV2V.add(0.0);
                continue;
            } else {
                // 卸载到其他车辆
                // 当前任务的传输大小（bit）
                int taskSize = currTask.getS();

                // TODO: 计算传输速率变量
                double rate4v2v = getTransRateV2O(vehicleOfCurrTask, objOffloadVehicle);

                // 计算当前任务的传输时间
                double currTaskTransTime = taskSize * unloadRatioRemote / rate4v2v;

                // 记录 currTask 传输时间
                taskUplinkTimeMap2Vehicle.put(i, currTaskTransTime);
            }

        }
    }

    /**
     * 计算 任务车辆 --> 目标车辆 的传输速率
     */
    public static double getTransRateV2O(int taskVehicleID, int objVehicleID) {
        int vehicleNums = gainChannelVehicles.length;
        // double widthV2O = Constants.WIDTH_CHANNEL_VEHICLE / (double) vehicleNums;
        double widthV2O = Constants.WIDTH_CHANNEL_VEHICLE;

        double gainChannelV2O = gainChannelVehicles[taskVehicleID];
        Long powerTransVehicle = Constants.POWER_TRANS_VEHICLE;

        Long powerNoise = Constants.POWER_NOISE;
        double A0 = Constants.A0;
        double lengthV2O = 1.0;  // 车辆间的距离
        lengthV2O = getTransDistanceV2O(taskVehicleID, objVehicleID);

        double transRateV2O
                = widthV2O * Math.log(1 + (powerTransVehicle * gainChannelV2O) / powerNoise
                + A0 * Math.pow(lengthV2O, -2));

        return transRateV2O;
    }

    /**
     * 计算任务车辆和计算车辆之间的距离
     *
     * @param taskVehicleID
     * @param objVehicleID
     * @return
     */
    public static double getTransDistanceV2O(int taskVehicleID, int objVehicleID) {
        double dis = 1.0;
        // 计算规则：找到 ＞当前车辆信道增益的车辆

        return dis;
    }


    /**
     * 计算task执行能耗Local
     *
     * @param taskList
     * @param unloadArr
     * @param unloadRatioArr
     * @param resAllocArrLocal
     * @param taskExecEnergyMapLocal
     */
    public static void calculateTaskExecEnergyLocal(List<Task> taskList,
                                                    List<Integer> unloadArr,
                                                    List<Double> unloadRatioArr,
                                                    List<Integer> resAllocArrLocal,
                                                    Map<Integer, Double> taskExecEnergyMapLocal) {
        taskExecEnergyMapLocal.clear();

        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            // Remote 卸载的比例
            double unloadRatioRemote = 1.0 - unloadRatioArr.get(i);

            // task 参数
            Task currTask = taskList.get(i);
            int currTaskSize = currTask.getS();
            float currTaskC = currTask.getC();

            // 分配频率
            int freqAlloc4task = resAllocArrLocal.get(i);

            double currTaskExecEnergy = 0.0;

            // 功率参数
            double coefficientPowerVehicle = Constants.COEFFICIENT_POWER_VEHICLE;
            double coefficientPowerRSU = Constants.COEFFICIENT_POWER_RSU;
            double coefficientPowerCloud = Constants.COEFFICIENT_POWER_CLOUD;

            // 卸载比例 * size
            currTaskSize = (int) (unloadRatioRemote * currTaskSize);

            if (unloadArr.get(i) > 0) {
                // vehicle
                currTaskExecEnergy
                        = coefficientPowerVehicle * currTaskSize * currTaskC * Math.pow(freqAlloc4task, 2);
            } else if (unloadArr.get(i) == 0) {
                // RSU
                currTaskExecEnergy
                        = coefficientPowerRSU * currTaskSize * currTaskC * Math.pow(freqAlloc4task, 2);
            } else if (unloadArr.get(i) == -1) {
                // cloud
                currTaskExecEnergy
                        = coefficientPowerCloud * currTaskSize * currTaskC * Math.pow(freqAlloc4task, 2);
            }

            // 记录任务i能耗
            taskExecEnergyMapLocal.put(i, currTaskExecEnergy);
        }
    }


    /**
     * 计算task执行能耗 Remote
     *
     * @param taskList
     * @param unloadArr
     * @param unloadRatioArr
     * @param resAllocArrRemote
     * @param taskExecEnergyMapRemote
     */
    public static void calculateTaskExecEnergyRemote(List<Task> taskList,
                                                     List<Integer> unloadArr,
                                                     List<Double> unloadRatioArr,
                                                     List<Integer> resAllocArrRemote,
                                                     Map<Integer, Double> taskExecEnergyMapRemote) {
        taskExecEnergyMapRemote.clear();

        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            // Remote 卸载的比例
            double unloadRatioRemote = unloadRatioArr.get(i);

            // task 参数
            Task currTask = taskList.get(i);
            int currTaskSize = currTask.getS();
            float currTaskC = currTask.getC();

            int freqAlloc4task = resAllocArrRemote.get(i);

            double currTaskExecEnergy = 0.0;

            double coefficientPowerVehicle = Constants.COEFFICIENT_POWER_VEHICLE;
            double coefficientPowerRSU = Constants.COEFFICIENT_POWER_RSU;
            double coefficientPowerCloud = Constants.COEFFICIENT_POWER_CLOUD;

            // 卸载比例
            currTaskSize = (int) (unloadRatioRemote * currTaskSize);
            if (unloadArr.get(i) > 0) {
                // vehicle
                currTaskExecEnergy
                        = coefficientPowerVehicle * currTaskSize * currTaskC * Math.pow(freqAlloc4task, 2);
            } else if (unloadArr.get(i) == 0) {
                // RSU
                currTaskExecEnergy
                        = coefficientPowerRSU * currTaskSize * currTaskC * Math.pow(freqAlloc4task, 2);
            } else if (unloadArr.get(i) == -1) {
                // cloud
                currTaskExecEnergy
                        = coefficientPowerCloud * currTaskSize * currTaskC * Math.pow(freqAlloc4task, 2);
            }

            // 记录任务 i 能耗
            taskExecEnergyMapRemote.put(i, currTaskExecEnergy);
        }
    }


    /**
     * 计算task传输到RSU的能耗
     *
     * @param taskList
     * @param unloadArr
     * @param unloadRatioArr
     * @param resAllocArr
     * @param taskUplinkTimeMap2RSU
     * @param taskUplinkEnergyMap2RSU
     */
    public static void calculateTaskUplinkEnergy2RSU(List<Task> taskList,
                                                     List<Integer> unloadArr,
                                                     List<Double> unloadRatioArr,
                                                     List<Integer> resAllocArr,
                                                     Map<Integer, Double> taskUplinkTimeMap2RSU,
                                                     Map<Integer, Double> taskUplinkEnergyMap2RSU) {
        taskUplinkEnergyMap2RSU.clear();

        // 计算时间 ====> taskUplinkTimeList2RSU
        calculateTaskTransTime4Uplink2RSU(taskList, unloadArr, unloadRatioArr, taskUplinkTimeMap2RSU);

        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            if (unloadArr.get(i) > 0 || !taskUplinkTimeMap2RSU.containsKey(i)) {
                // taskUplinkEnergyList2RSU.add(0.0);
                continue;
            } else {
                // Task currTask = taskList.get(i);
                // int currTaskSize = currTask.getS();
                // float currTaskC = currTask.getC();
                // int freqAlloc4task = resArr.get(i);

                Long powerTransVehicle = Constants.POWER_TRANS_VEHICLE; // 功率
                // 上传时间
                double currTaskUplinkTime = taskUplinkTimeMap2RSU.get(i);

                // 计算能耗
                double currTaskUplinkEnergy = powerTransVehicle * currTaskUplinkTime;

                // 记录能耗
                taskUplinkEnergyMap2RSU.put(i, currTaskUplinkEnergy);
            }
        }
    }


    /**
     * 计算task传输到Cloud的上行能耗
     *
     * @param taskList
     * @param unloadArr
     * @param unloadRatioArr
     * @param resAllocArr
     * @param taskUplinkTimeMap2Cloud
     * @param taskUplinkEnergyMap2Cloud
     */
    public static void calculateTaskUplinkEnergy2Cloud(List<Task> taskList,
                                                       List<Integer> unloadArr,
                                                       List<Double> unloadRatioArr,
                                                       List<Integer> resAllocArr,
                                                       Map<Integer, Double> taskUplinkTimeMap2Cloud,
                                                       Map<Integer, Double> taskUplinkEnergyMap2Cloud) {
        taskUplinkEnergyMap2Cloud.clear();

        // 计算时间 ====> taskUplinkTimeList2Cloud
        calculateTaskTransTime4UplinkR2C(taskList, unloadArr, unloadRatioArr, taskUplinkTimeMap2Cloud);

        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            if (unloadArr.get(i) >= 0 || !taskUplinkTimeMap2Cloud.containsKey(i)) {
                // taskUplinkEnergyList2Cloud.add(0.0);
                continue;
            } else {
                Long powerTransRSU = Constants.POWER_TRANS_RSU; // 功率
                // 上传时间
                double currTaskUplinkTime = taskUplinkTimeMap2Cloud.get(i);

                // 计算能耗
                double currTaskUplinkEnergy = powerTransRSU * currTaskUplinkTime;

                // 记录任务 i 的能耗
                taskUplinkEnergyMap2Cloud.put(i, currTaskUplinkEnergy);
            }
        }
    }


    /**
     * 计算任务i的车辆间传输能耗
     *
     * @param taskList
     * @param unloadArr
     * @param unloadRatioArr
     * @param taskUplinkTimeMapV2V  ： 任务传输时间V2V  <taskID, time>
     * @param taskTransEnergyMapV2V
     */
    public static void calculateTaskTransEnergy4V2V(List<Task> taskList,
                                                    List<Integer> unloadArr,
                                                    List<Double> unloadRatioArr,
                                                    Map<Integer, Double> taskUplinkTimeMapV2V,
                                                    Map<Integer, Double> taskTransEnergyMapV2V) {
        taskTransEnergyMapV2V.clear();

        // 时间计算 ====> taskUplinkTimeListV2V
        calculateTaskTransTime4V2V(taskList, unloadArr, unloadRatioArr, taskUplinkTimeMapV2V);

        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            if (unloadArr.get(i) <= 0 || !taskUplinkTimeMapV2V.containsKey(i)) {
                // taskTransEnergyListV2V.add(0.0);
                continue;
            } else {

                Task currTask = taskList.get(i);
                // 当前任务所属车辆的id
                int currTaskVehicleID = currTask.getVehicleID();
                int objVehicleID = unloadArr.get(i);

                if (currTaskVehicleID == objVehicleID) {
                    // taskTransEnergyListV2V.add(0.0);
                    continue;
                } else {
                    Long powerTransVehicle = Constants.POWER_TRANS_VEHICLE;
                    Double transTime = taskUplinkTimeMapV2V.get(i);
                    Double energy = powerTransVehicle.doubleValue() * transTime;

                    // 记录任务i的车辆间传输能耗
                    taskTransEnergyMapV2V.put(i, energy);
                }
            }
        }
    }


}
