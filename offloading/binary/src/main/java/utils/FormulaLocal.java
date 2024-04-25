package utils;

import com.sun.xml.bind.v2.runtime.output.DOMOutput;
import com.sun.xml.bind.v2.runtime.reflect.opt.Const;
import config.InitFrame;
import entity.Task;
import entity.Vehicle;
import enums.Constants;
import enums.TaskPolicy;
import lombok.extern.slf4j.Slf4j;

import java.util.*;

@Slf4j(topic = "Formula_Local")
public class FormulaLocal {

    // 车辆的信道增益
    // 车辆编号从1开始，gainChannelVehicles[0] 跳过
    static double[] gainChannelVehicles;
    // List<Double> gainChannelVehiclesList = new ArrayList<>();


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

    public static List<Double> getUss4TaskListLocalOnly(List<Task> taskList,
                                                        List<Vehicle> vList,
                                                        List<Integer> unloadArr,
                                                        List<Integer> freqAllocArr) {
        List<Double> taskUSSList = new ArrayList<>();

        int len = taskList.size();
        // log.info("========================= 计算任务满意度List =========================");
        taskUSSList.clear();

        // 计算任务的花费时间
        Map<Integer, Double> taskCostTimeMap = new HashMap<>();
        calculateTaskExecTimeLocalOnly(taskList, vList, unloadArr, freqAllocArr, taskCostTimeMap);
        // 得到task最大优先级
        int taskMaxPrior = getTaskMaxPrior(taskList);
        // 遍历每个任务
        for (int taskIndex = 0; taskIndex < len; taskIndex++) {
            // 当前任务花费时间
            double currTaskCostTime = taskCostTimeMap.get(taskIndex);
            // 当前任务
            Task currTask = taskList.get(taskIndex);
            // 属性
            double currTaskDeadline = currTask.getD() / 100.0;
            int currTaskFactor = currTask.getFactor();
            double currTaskMaxDeadline = currTaskFactor * currTaskDeadline;
            int currTaskPrior = currTask.getP();
            // 是否在最大截止期限内处理完成
            if (currTaskCostTime <= currTaskMaxDeadline) {
                double exp
                        = Math.max(0, (currTaskCostTime - currTaskDeadline))
                        / currTaskMaxDeadline * (currTaskPrior / taskMaxPrior);
                // 用户满意度计算阈值 θ
                double theta = Constants.USS_THRESHOLD;
                double currTaskUSS = theta - Math.pow(theta, exp);
                taskUSSList.add(currTaskUSS);
            } else {
                taskUSSList.add(0.0);
            }
        }
        return taskUSSList;
    }

    public static List<Double> getEnergy4TaskListLocalOnly(List<Task> taskList,
                                                           List<Integer> unloadArr,
                                                           List<Integer> freqAllocArr) {
        List<Double> taskEnergyListLocalOnly = new ArrayList<>();

        List<Double> taskExecTimeList = new ArrayList<>();
        calculateTaskExecTimeNoSort(taskList, unloadArr, freqAllocArr, taskExecTimeList);

        int size = taskList.size();

        for (int i = 0; i < size; i++) {
            Task currTask = taskList.get(i);
            int currTaskSize = currTask.getS();
            float currTaskC = currTask.getC();

            int freqAlloc4task = freqAllocArr.get(i);

            double currTaskExecEnergy = 0.0;

            double coefficientPowerVehicle = Constants.COEFFICIENT_POWER_VEHICLE;

            currTaskExecEnergy = coefficientPowerVehicle * currTaskSize * currTaskC * Math.pow(freqAlloc4task, 2);

            taskEnergyListLocalOnly.add(currTaskExecEnergy);
        }

        return taskEnergyListLocalOnly;
    }

    public static void calculateTaskExecTimeNoSort(List<Task> taskList,
                                                   List<Integer> unloadArr,
                                                   List<Integer> freqAllocArr,
                                                   List<Double> taskExecTimeList) {
        int size = taskList.size();

        for (int i = 0; i < size; i++) {
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
            taskExecTimeList.add(currTaskExecTime);
        }

    }

    // 得到taskList的ussList
    public static List<Double> getUSS4TaskList(List<Task> taskList,
                                               List<Integer> unloadArr,
                                               List<Integer> resAllocArr) {
        List<Double> taskUSSList = new ArrayList<>();
        calculateUSS4TaskList(taskList, unloadArr, resAllocArr, taskUSSList);
        return taskUSSList;
    }

    // 计算任务list满意度
    public static void calculateUSS4TaskList(List<Task> taskList,
                                             List<Integer> unloadArr,
                                             List<Integer> resAllocArr,
                                             List<Double> taskUSSList) {
        int len = taskList.size();
        // log.info("========================= 计算任务满意度List =========================");
        taskUSSList.clear();

        // 计算任务的花费时间
        List<Double> taskCostTimeList = new ArrayList<>();
        calculateTaskCostTime(taskList, unloadArr, resAllocArr, taskCostTimeList);
        // 得到task最大优先级
        int taskMaxPrior = getTaskMaxPrior(taskList);

        // 遍历每个任务
        for (int taskIndex = 0; taskIndex < len; taskIndex++) {
            // 当前任务花费时间
            double currTaskCostTime = taskCostTimeList.get(taskIndex);
            // 当前任务
            Task currTask = taskList.get(taskIndex);
            // 属性
            double currTaskDeadline = currTask.getD() / 100.0;
            int currTaskFactor = currTask.getFactor();
            double currTaskMaxDeadline = currTaskFactor * currTaskDeadline;
            int currTaskPrior = currTask.getP();
            // 是否在最大截止期限内处理完成
            if (currTaskCostTime <= currTaskMaxDeadline) {
                double exp = Math.max(0, (currTaskCostTime - currTaskDeadline)) / currTaskMaxDeadline * currTaskPrior / taskMaxPrior;
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
                                                  List<Integer> resAllocArr) {
        List<Double> taskEnergyList = new ArrayList<>();
        calculateEnergy4TaskList(taskList, unloadArr, resAllocArr, taskEnergyList);
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
                                                List<Integer> resAllocArr,
                                                List<Double> taskEnergyList) {

        // log.info("==================== 计算任务总能耗 ===================");
        taskEnergyList.clear();

        // 更新 energy list
        // calculateXXX()
        List<Double> taskExecEnergyList = new ArrayList<>();
        List<Double> taskUplinkEnergyV2RList = new ArrayList<>();
        List<Double> taskUplinkEnergyR2CList = new ArrayList<>();
        List<Double> taskUplinkEnergyV2VList = new ArrayList<>();
        // 计算task执行能耗
        calculateTaskExecEnergy(taskList, unloadArr, resAllocArr, taskExecEnergyList);

        // todo：前提，计算时间【公共部分提取】
        List<Double> taskUplinkTimeV2VList = new ArrayList<>();
        List<Double> taskUplinkTimeV2RList = new ArrayList<>();
        List<Double> taskUplinkTimeR2CList = new ArrayList<>();
        List<Double> taskExecTimeList = new ArrayList<>();
        // 计算时间
        calculateTaskTransTime4Uplink2RSU(taskList, unloadArr, taskUplinkTimeV2RList);
        calculateTaskTransTime4UplinkR2C(taskList, unloadArr, taskUplinkTimeR2CList);
        calculateTaskTransTime4V2V(taskList, unloadArr, taskUplinkTimeV2VList);
        // 计算能耗
        // calculateTaskUplinkEnergy2RSU(taskList, resAllocArr, );
        calculateTaskUplinkEnergy2RSU(taskList, unloadArr, resAllocArr, taskUplinkTimeV2RList, taskUplinkEnergyV2RList);
        calculateTaskUplinkEnergy2Cloud(taskList, unloadArr, resAllocArr, taskUplinkTimeR2CList, taskUplinkEnergyR2CList);
        calculateTaskTransEnergy4V2V(taskList, unloadArr, taskUplinkTimeV2VList, taskUplinkEnergyV2VList);

        int len = taskList.size();

        for (int i = 0; i < len; i++) {
            double currTaskCostEnergy = 0.0;

            int unloadObjNode = unloadArr.get(i);

            Task currTask = taskList.get(i);
            int currTaskVehicleID = currTask.getVehicleID();

            /*
            if(unloadObjNode == -1) {
                // cloud
                currTaskCostEnergy = taskExecEnergyList.get(i) + taskUplinkEnergyV2RList.get(i) + taskUplinkEnergyR2CList.get(i);
            } else if (unloadObjNode == 0) {
                // rsu
                currTaskCostEnergy = taskExecEnergyList.get(i) + taskUplinkEnergyV2RList.get(i);
            } else if (unloadObjNode == currTaskVehicleID) {
                // local
                currTaskCostEnergy = taskExecEnergyList.get(i);
            } else {
                // other vehicles
                currTaskCostEnergy = taskExecEnergyList.get(i) + taskUplinkEnergyV2VList.get(i);
            }
            */

            currTaskCostEnergy = taskExecEnergyList.get(i) + taskUplinkEnergyV2RList.get(i)
                    + taskUplinkEnergyR2CList.get(i) + taskUplinkEnergyV2VList.get(i);

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
                                             List<Integer> resAllocArr,
                                             List<Double> taskCostTimeList) {
        // log.info("========================= 计算任务花费时间 =========================");
        // res
        taskCostTimeList.clear();

        // note: 更新 【time list】
        List<Double> taskExecTimeList = new ArrayList<>();
        List<Double> taskUplinkTimeV2RList = new ArrayList<>();
        List<Double> taskUplinkTimeR2CList = new ArrayList<>();
        List<Double> taskUplinkTimeV2VList = new ArrayList<>();
        calculateTaskExecTime(taskList, unloadArr, resAllocArr, taskExecTimeList);
        calculateTaskTransTime4Uplink2RSU(taskList, unloadArr, taskUplinkTimeV2RList);
        calculateTaskTransTime4UplinkR2C(taskList, unloadArr, taskUplinkTimeR2CList);
        calculateTaskTransTime4V2V(taskList, unloadArr, taskUplinkTimeV2VList);

        int len = taskList.size();

        for (int i = 0; i < len; i++) {
            Task currTask = taskList.get(i);
            // 任务所述车辆
            int vehicleNum4Task = currTask.getVehicleID();

            double currTaskCostTime = 0.0;

            int unloadObjNode = unloadArr.get(i);
            if (unloadObjNode == -1) {
                // 卸载到 cloud
                currTaskCostTime = taskExecTimeList.get(i) + taskUplinkTimeV2RList.get(i) + taskUplinkTimeR2CList.get(i);
            } else if (unloadObjNode == 0) {
                // 卸载到 rsu
                currTaskCostTime = taskExecTimeList.get(i) + taskUplinkTimeV2RList.get(i);
            } else if (unloadObjNode == vehicleNum4Task) {
                // 卸载到 local
                currTaskCostTime = taskExecTimeList.get(i);
            } else {
                // 卸载到 目标车辆
                currTaskCostTime = taskExecTimeList.get(i) + taskUplinkTimeV2VList.get(i);
            }

            taskCostTimeList.add(currTaskCostTime);
        }

    }


    /**
     * 计算任务执行时间  ---- 仅仅 Local 处理的情况
     * 【test --> ok】
     *
     * @param taskList
     * @param unloadArr
     * @param resArr
     * @param taskExecTimeList
     */
    public static void calculateTaskExecTime(List<Task> taskList,
                                             List<Integer> unloadArr,
                                             List<Integer> resArr,
                                             List<Double> taskExecTimeList) {
        // clear 执行时间 list
        taskExecTimeList.clear();

        // 获取每个车辆的 taskIdList
        Map<Integer, List<Integer>> currVehicleTasksIdMap = new HashMap<>();

        int len = taskList.size();

        for (int i = 0; i < len; i++) {
            // 当前任务
            Task currTask = taskList.get(i);
            int taskSize = currTask.getS();
            float processDensity = currTask.getC();

            // currTask 的 计算需求量
            double processCost = taskSize * processDensity;
            // 分配给 currTask 的计算资源量
            int resAlloc2currTask = resArr.get(i);

            // 计算执行时间
            double currTaskExecTime = processCost / (double) resAlloc2currTask;

            // 加入 taskExecTimeList
            taskExecTimeList.add(currTaskExecTime);
        }
    }

    public static List<Double> getTaskCostTimeLocalOnly(List<Task> taskList,
                                                      List<Vehicle> vehicleList,
                                                      List<Integer> unloadArr,
                                                      List<Integer> freqAllocArr) {
        List<Double> taskCostTimeLocalOnly = new ArrayList<>();
        Map<Integer, Double> taskExecTimeMap = new HashMap<>();
        calculateTaskExecTimeLocalOnly(taskList, vehicleList, unloadArr, freqAllocArr,
                taskExecTimeMap);

        int size = taskList.size();
        for (int i = 0; i < size; i++) {
            taskCostTimeLocalOnly.add(taskExecTimeMap.get(i));
        }

        for (int i = 0; i < size; i++) {
            taskCostTimeLocalOnly.set(i, FormatData.getEffectiveValue4Digit(taskCostTimeLocalOnly.get(i), 5));
        }

        return taskCostTimeLocalOnly;
    }

    // NOTE ： 任务仅在本地处理的情况
    public static void calculateTaskExecTimeLocalOnly(List<Task> taskList,
                                                      List<Vehicle> vehicleList,
                                                      List<Integer> unloadArr,
                                                      List<Integer> freqAllocArr,
                                                      Map<Integer, Double> taskExecTimeMap) {
        // clear 执行时间 list
        taskExecTimeMap.clear();

        // 获取每个车辆的 taskIdList
        Map<Integer, List<Integer>> currVehicleTasksIdMap = new HashMap<>();

        int vehicleNums = vehicleList.size();

        int revise_select = TaskPolicy.TASK_LIST_SORT_RULE;

        for (int i = 1; i < vehicleNums + 1; i++) {
            // 当前编号为 i 车辆的 任务集
            List<Integer> currVehicleTaskIDsList = new ArrayList<>();
            currVehicleTaskIDsList = getVehicle2TaskIdList(i, taskList);

            if (revise_select == 1) {
                // 优先级排序（降序）
                SortRules.sortByTaskPrior(currVehicleTaskIDsList, taskList);
            } else if (revise_select == 2) {
                // 计算量排序（升序）
                SortRules.sortMinComputationFirstSequence(currVehicleTaskIDsList, taskList);
            } else if (revise_select == 3) {
                // deadline 排序（升序）
                SortRules.sortMinDeadLineFirstSequence(currVehicleTaskIDsList, taskList);
            }

            int currVehicleFreqRemain = (int) vehicleList.get(i - 1).getFreqRemain();

            // 当前车辆的任务集所需要的freq
            List<Integer> currVehicleTasksNeedFreqList = new ArrayList<>();
            for (int j = 0; j < currVehicleTaskIDsList.size(); j++) {
                int tempFreqNeed = freqAllocArr.get(currVehicleTaskIDsList.get(j));
                currVehicleTasksNeedFreqList.add(tempFreqNeed);
            }

            // 不排序情况下，当前车辆任务集的执行时间
            List<Double> tempTasksExecTime = new ArrayList<>();
            for (int j = 0; j < currVehicleTaskIDsList.size(); j++) {
                int currTaskId = currVehicleTaskIDsList.get(j);

                // 分配给 currTask 的计算资源量
                int currTaskFreqAlloc = freqAllocArr.get(currTaskId);
                // 当前任务
                Task currTask = taskList.get(currTaskId);
                int taskSize = currTask.getS();
                float processDensity = currTask.getC();
                // currTask 的 计算需求量
                double processCost = taskSize * processDensity;

                // 计算执行时间
                double currTaskExecTime = processCost / (double) currTaskFreqAlloc;

                tempTasksExecTime.add(currTaskExecTime);
            }

            // NOTE： freq 不足分配时，需要加上排序靠前任务的 execTime
            int offTasksFreqNeed = 0;
            int size = currVehicleTaskIDsList.size();
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
                int tempTaskID = currVehicleTaskIDsList.get(task_index_need);
                int tempFreqNeed = freqAllocArr.get(tempTaskID);

                if (currVehicleFreqRemain >= tempFreqNeed) {
                    // 车辆 freqRemain 减少
                    currVehicleFreqRemain -= tempFreqNeed;
                    taskExecTimeMap.put(tempTaskID,
                            needPlusTime + tempTasksExecTime.get(task_index_need));

                    // 当前Node已执行完的任务，及其时间
                    taskAlreadyExecIdsList.add(tempTaskID);
                    taskAlreadyExecTimeMap.put(tempTaskID, needPlusTime + tempTasksExecTime.get(task_index_need));

                    // 待处理任务指针后移
                    ++task_index_need;
                } else {
                    // 当前处理N个任务，NodeFreqRemain不足处理下一个task时
                    if (task_index_already >= task_index_need) {
                        // 说明分配freq超出 当前NodeFreqRemain
                        freqAllocArr.set(tempTaskID, currVehicleFreqRemain);
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
                    currVehicleFreqRemain += freqAllocArr.get(minExecTimeTaskId);

                    // 下一次资源足够时，需要加上的时间
                    needPlusTime = Math.max(needPlusTime,
                            taskAlreadyExecTimeMap.get(minExecTimeTaskId));

                    // 移除已经算了时间的任务
                    // taskAlreadyExecIdsList.remove(minExecTimeTaskId);
                    // taskAlreadyExecIdsList.remove(task_index_already);
                    // taskAlreadyExecTimeMap.remove(task_index_already);
                    ++task_index_already;
                }
            }

        }

    }

    // 获取编号为 vehicleID 的任务
    public static List<Integer> getVehicle2TaskIdList(int vehicleID,
                                                      List<Task> taskList) {
        List<Integer> tasksIdList = new ArrayList<>();

        int len = taskList.size();

        for (int i = 0; i < len; i++) {
            int currTaskID = taskList.get(i).getTaskID();
            int currTask2VehicleID = taskList.get(i).getVehicleID();

            if (vehicleID == currTask2VehicleID) {
                tasksIdList.add(currTaskID);
            }
        }

        return tasksIdList;
    }

    /**
     * TODO: 当卸载节点freq不足以分配时，排序处理卸载到此节点的任务
     *
     * @param taskList
     * @param unloadArr
     * @param resArr
     * @param taskExecTimeList
     */
    public static void calculateTaskExecTimeBySort(List<Task> taskList,
                                                   List<Integer> unloadArr,
                                                   List<Integer> resArr,
                                                   List<Double> taskExecTimeList) {
        // clear 执行时间 list
        taskExecTimeList.clear();

        int len = taskList.size();

        // 获取卸载到每个节点的任务id
        Map<Integer, List<Integer>> nodeUnloadTaskIDsMap = new HashMap<>();
        for (int i = 0; i < len; i++) {
            int unloadNodeID = unloadArr.get(i);
            if (!nodeUnloadTaskIDsMap.containsKey(unloadNodeID)) {
                nodeUnloadTaskIDsMap.put(unloadNodeID, new ArrayList<>());
            }
            nodeUnloadTaskIDsMap.get(unloadNodeID).add(i);
        }

        for (int i = 0; i < len; i++) {
            // 当前任务
            Task currTask = taskList.get(i);
            int taskSize = currTask.getS();
            float processDensity = currTask.getC();

            // currTask 的 计算需求量
            double processCost = taskSize * processDensity;
            // 分配给 currTask 的计算资源量
            int resAlloc2currTask = resArr.get(i);

            // 计算执行时间
            double currTaskExecTime = processCost / (double) resAlloc2currTask;

            // 加入 taskExecTimeList
            taskExecTimeList.add(currTaskExecTime);
        }
    }

    /**
     * 计算车辆传输到RSU的任务上行传输时间
     *
     * @param taskMap
     * @param unloadArr
     * @param taskUplinkTimeMap
     */
    public static void calculateTaskTransTime4Uplink2RSU(Map<Integer, Task> taskMap,
                                                         List<Integer> unloadArr,
                                                         Map<Integer, Double> taskUplinkTimeMap) {
        taskUplinkTimeMap.clear();

        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            // 非卸载到 rsu 的任务，跳过
            if (unloadArr.get(i) != 0) continue;

            // 当前任务
            Task currTask = taskMap.get(i);

            int taskSize = currTask.getS();

            // 传输速率变量
            double rate4v2r = getTransRateV2R(currTask.getVehicleID(), unloadArr);

            double currTaskUplinkTime = taskSize / (double) rate4v2r;

            // 记录 currTask 传输时间
            taskUplinkTimeMap.put(i, currTaskUplinkTime);
        }

    }

    /**
     * 计算需传输到RSU任务的上行传输时间
     *
     * @param //                 taskMap / taskList
     * @param unloadArr
     * @param taskUplinkTimeList : 上行传输时间的结果list
     */
    public static void calculateTaskTransTime4Uplink2RSU(List<Task> taskList, /* Map<Integer, Task> taskMap,*/
                                                         List<Integer> unloadArr,
                                                         List<Double> taskUplinkTimeList) {
        // 清理
        taskUplinkTimeList.clear();

        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            // 不需要传输到 rsu 的任务
            if (unloadArr.get(i) != 0 && unloadArr.get(i) != -1) {
                taskUplinkTimeList.add(0.0);
            } else {
                // 当前任务
                Task currTask = taskList.get(i);

                int taskSize = currTask.getS();

                // 传输速率变量
                // double rate4v2r = getTransRateV2R(currTask.getVehicleID());
                double rate4v2r = new Random().nextDouble() + 1.0;

                double currTaskUplinkTime = taskSize / (double) rate4v2r;

                // 记录 currTask 传输时间
                taskUplinkTimeList.add(currTaskUplinkTime);
            }
        }
    }

    /**
     * 计算从RSU传输到cloud的上行传输时间
     *
     * @param taskList
     * @param unloadArr
     * @param taskUplinkTimeListR2C
     */
    public static void calculateTaskTransTime4UplinkR2C(List<Task> taskList,
                                                        List<Integer> unloadArr,
                                                        List<Double> taskUplinkTimeListR2C) {
        taskUplinkTimeListR2C.clear();

        // 传输速率变量
        double rate4r2c = getTransRateR2C();

        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            if (unloadArr.get(i) != -1) {
                taskUplinkTimeListR2C.add(0.0);
            } else {
                // 当前任务
                Task currTask = taskList.get(i);

                int taskSize = currTask.getS();
                // 计算当前任务的传输时间
                double currTaskUplinkTime = taskSize / (double) rate4r2c;

                // 记录 currTask 传输时间
                taskUplinkTimeListR2C.add(currTaskUplinkTime);
            }
        }

    }

    /**
     * 计算车辆间任务传输时间
     *
     * @param taskList
     * @param unloadArr
     * @param taskTransTimeListV2V
     */
    public static void calculateTaskTransTime4V2V(List<Task> taskList,
                                                  List<Integer> unloadArr,
                                                  List<Double> taskTransTimeListV2V) {
        taskTransTimeListV2V.clear();

        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            // 传输到 rsu 的任务，跳过
            // if(unloadArr.get(i) <= 0) continue;
            if (unloadArr.get(i) <= 0) {
                taskTransTimeListV2V.add(0.0);
                continue;
            }

            // 计算
            Task currTask = taskList.get(i);
            // currTask 所属车辆
            int vehicleOfCurrTask = currTask.getVehicleID();
            // currTask 卸载的目标车辆
            int objOffloadVehicle = unloadArr.get(i);
            // 卸载 local
            if (vehicleOfCurrTask == objOffloadVehicle) {
                taskTransTimeListV2V.add(0.0);
            } else {
                // 卸载到其他车辆
                // 当前任务的传输大小（bit）
                int taskSize = currTask.getS();

                // TODO: 计算传输速率变量
                double rate4v2v = getTransRateV2O(vehicleOfCurrTask, objOffloadVehicle);

                // 计算当前任务的传输时间
                double currTaskTransTime = taskSize / rate4v2v;

                // 记录 currTask 传输时间
                taskTransTimeListV2V.add(currTaskTransTime);
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
     * 计算Cloud到RSU的传输速率
     */
    public static double getTransRateC2R() {
        double widthC2R = Constants.WIDTH_CHANNEL_CLOUD;
        Long gainChannelC2R = Constants.GAIN_CHANNEL_C2R;
        Long powerTransCloud = Constants.POWER_TRANS_CLOUD;
        Long powerNoise = Constants.POWER_NOISE;

        double transRateC2R = widthC2R * Math.log(1 + (powerTransCloud * gainChannelC2R) / powerNoise);

        return transRateC2R;
    }

    /**
     * 计算vehicle到rsu的传输速率
     */
    public static double getTransRateV2R(int vehicleID, List<Integer> unloadArr) {

        // 需要传输task到rsu的车辆数目
        int vehicleNums2RSU = 0;
        int len = unloadArr.size();
        for (int i = 0; i < len; i++) {
            if (unloadArr.get(i) <= 0) ++vehicleNums2RSU;
        }

        double widthC2R = Constants.WIDTH_CHANNEL_RSU / (double) vehicleNums2RSU;

        double gainChannelV2R = gainChannelVehicles[vehicleID];
        Long powerTransVehicle = Constants.POWER_TRANS_VEHICLE;

        Long powerNoise = Constants.POWER_NOISE;

        // todo:计算来自其他车辆的无线干扰
        double otherVehiclesNoise = 1.0;

        double transRateV2R = widthC2R * Math.log(1 + (powerTransVehicle * gainChannelV2R) / powerNoise + otherVehiclesNoise);

        return transRateV2R;
    }

    /**
     * 计算 rsu --> vehicle 的传输速率
     */
    public static double getTransRateR2V(int vehicleID) {
        int vehicleNums = Constants.VEHICLE_NUMS;
        double widthR2C = Constants.WIDTH_CHANNEL_RSU / (double) vehicleNums;

        double gainChannelR2V = Constants.GAIN_CHANNEL_R2V;
        Long powerTransRSU = Constants.POWER_TRANS_RSU;

        Long powerNoise = Constants.POWER_NOISE;
        double otherVehiclesNoise = 1.0;  // 计算来自其他车辆的无线干扰

        double transRateR2V = widthR2C * Math.log(1 + (powerTransRSU * gainChannelR2V) / powerNoise + otherVehiclesNoise);

        return transRateR2V;
    }

    public static double getTransOtherVehicleNoise4V2R(int vehicleID) {
        double otherVehiclesNoiseV2R = 0.0;
        // 计算规则：找到 ＞当前车辆信道增益的车辆

        return otherVehiclesNoiseV2R;
    }

    /**
     * 计算 任务车辆 --> 目标车辆 的传输速率
     */
    public static double getTransRateV2O(int taskVehicleID, int objVehicleID) {
        int vehicleNums = gainChannelVehicles.length;
        double widthV2O = Constants.WIDTH_CHANNEL_VEHICLE / (double) vehicleNums;

        double gainChannelV2O = gainChannelVehicles[taskVehicleID];
        Long powerTransVehicle = Constants.POWER_TRANS_VEHICLE;

        Long powerNoise = Constants.POWER_NOISE;
        double A0 = Constants.A0;
        double lengthV2O = 1.0;  // 车辆间的距离
        lengthV2O = getTransDistanceV2O(taskVehicleID, objVehicleID);

        double transRateV2O = widthV2O * Math.log(1 + (powerTransVehicle * gainChannelV2O) / powerNoise + A0 * Math.pow(lengthV2O, -2));

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
     * 计算任务执行能耗
     *
     * @param taskList
     * @param unloadArr
     * @param resArr
     * @param taskExecEnergyList
     */
    public static void calculateTaskExecEnergy(List<Task> taskList,
                                               List<Integer> unloadArr,
                                               List<Integer> resArr,
                                               List<Double> taskExecEnergyList) {
        taskExecEnergyList.clear();

        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            Task currTask = taskList.get(i);
            int currTaskSize = currTask.getS();
            float currTaskC = currTask.getC();

            int freqAlloc4task = resArr.get(i);

            double currTaskExecEnergy = 0.0;

            double coefficientPowerVehicle = Constants.COEFFICIENT_POWER_VEHICLE;
            double coefficientPowerRSU = Constants.COEFFICIENT_POWER_RSU;
            double coefficientPowerCloud = Constants.COEFFICIENT_POWER_CLOUD;

            if (unloadArr.get(i) > 0) {
                // vehicle
                currTaskExecEnergy = coefficientPowerVehicle * currTaskSize * currTaskC * Math.pow(freqAlloc4task, 2);
            } else if (unloadArr.get(i) == 0) {
                // RSU
                currTaskExecEnergy = coefficientPowerRSU * currTaskSize * currTaskC * Math.pow(freqAlloc4task, 2);
            } else if (unloadArr.get(i) == -1) {
                // cloud
                currTaskExecEnergy = coefficientPowerCloud * currTaskSize * currTaskC * Math.pow(freqAlloc4task, 2);
            }

            taskExecEnergyList.add(currTaskExecEnergy);
        }
    }


    /**
     * 计算任务上行传输到RSU的能耗
     *
     * @param taskList
     * @param unloadArr
     * @param resArr
     * @param taskUplinkEnergyList2RSU
     */
    public static void calculateTaskUplinkEnergy2RSU(List<Task> taskList,
                                                     List<Integer> unloadArr,
                                                     List<Integer> resArr,
                                                     List<Double> taskUplinkTimeList2RSU,
                                                     List<Double> taskUplinkEnergyList2RSU) {
        taskUplinkEnergyList2RSU.clear();

        // 计算时间 ====> taskUplinkTimeList2RSU
        calculateTaskTransTime4Uplink2RSU(taskList, unloadArr, taskUplinkTimeList2RSU);

        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            if (unloadArr.get(i) > 0) {
                taskUplinkEnergyList2RSU.add(0.0);
            } else {
                // Task currTask = taskList.get(i);
                // int currTaskSize = currTask.getS();
                // float currTaskC = currTask.getC();
                // int freqAlloc4task = resArr.get(i);

                Long powerTransVehicle = Constants.POWER_TRANS_VEHICLE; // 功率
                // 上传时间
                double currTaskUplinkTime = taskUplinkTimeList2RSU.get(i);

                // 计算能耗
                double currTaskUplinkEnergy = powerTransVehicle * currTaskUplinkTime;

                taskUplinkEnergyList2RSU.add(currTaskUplinkEnergy);
            }
        }

    }


    /**
     * 计算任务传输到cloud的上行能耗
     *
     * @param taskList
     * @param unloadArr
     * @param resArr
     * @param taskUplinkTimeList2Cloud
     * @param taskUplinkEnergyList2Cloud
     */
    public static void calculateTaskUplinkEnergy2Cloud(List<Task> taskList,
                                                       List<Integer> unloadArr,
                                                       List<Integer> resArr,
                                                       List<Double> taskUplinkTimeList2Cloud,
                                                       List<Double> taskUplinkEnergyList2Cloud) {
        taskUplinkEnergyList2Cloud.clear();

        // 计算时间 ====> taskUplinkTimeList2Cloud
        calculateTaskTransTime4UplinkR2C(taskList, unloadArr, taskUplinkTimeList2Cloud);

        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            if (unloadArr.get(i) >= 0) {
                taskUplinkEnergyList2Cloud.add(0.0);
            } else {
                Long powerTransRSU = Constants.POWER_TRANS_RSU; // 功率
                // 上传时间
                double currTaskUplinkTime = taskUplinkTimeList2Cloud.get(i);

                // 计算能耗
                double currTaskUplinkEnergy = powerTransRSU * currTaskUplinkTime;

                taskUplinkEnergyList2Cloud.add(currTaskUplinkEnergy);
            }
        }
    }


    /**
     * 计算车辆间任务传输能耗
     *
     * @param taskList
     * @param unloadArr
     * @param taskUplinkTimeListV2V
     * @param taskTransEnergyListV2V
     */
    public static void calculateTaskTransEnergy4V2V(List<Task> taskList,
                                                    List<Integer> unloadArr,
                                                    List<Double> taskUplinkTimeListV2V,
                                                    List<Double> taskTransEnergyListV2V) {
        taskTransEnergyListV2V.clear();

        // 时间计算 ====> taskUplinkTimeListV2V
        calculateTaskTransTime4V2V(taskList, unloadArr, taskUplinkTimeListV2V);

        int len = unloadArr.size();

        for (int i = 0; i < len; i++) {
            if (unloadArr.get(i) <= 0) {
                taskTransEnergyListV2V.add(0.0);
            } else {

                Task currTask = taskList.get(i);
                // 当前任务所属车辆的id
                int currTaskVehicleID = currTask.getVehicleID();
                int objVehicleID = unloadArr.get(i);

                if (currTaskVehicleID == objVehicleID) {
                    taskTransEnergyListV2V.add(0.0);
                } else {
                    Long powerTransVehicle = Constants.POWER_TRANS_VEHICLE;
                    Double transTime = taskUplinkTimeListV2V.get(i);
                    Double energy = powerTransVehicle.doubleValue() * transTime;
                    taskTransEnergyListV2V.add(energy);
                }
            }
        }

    }

}
