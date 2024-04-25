package utils;

import entity.Task;
import entity.Vehicle;
import enums.Constants;
import enums.TaskPolicy;

import java.util.*;

public class SortRules {
    /**
     * Sort Rule
     *
     * @param taskIdsListOfNode : 当前节点待卸载的任务id-list
     * @param taskList
     */
    public static void sortListByRules(List<Integer> taskIdsListOfNode,
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

    /**
     *
     * @param taskIdsListOfNode  当前节点待卸载的任务id-list
     * @param taskList
     * @param taskCostTimeList
     * @param freqAllocList
     * @param currVehicleFreqRemain
     */
    public static void sortListByRules(List<Integer> taskIdsListOfNode,
                                       List<Task> taskList,
                                       List<Double> taskCostTimeList,
                                       List<Integer> freqAllocList,
                                       double currVehicleFreqRemain) {

        int revise_select = TaskPolicy.TASK_LIST_SORT_RULE;

        if (revise_select == 1) {
            // 最大完成时间优先
            SortRules.sortTaskByMAXCTF(taskIdsListOfNode, taskCostTimeList);
        } else if (revise_select == 2) {
            // 最大计算量优先
            SortRules.sortTaskByMAXCLF(taskIdsListOfNode, taskList);
        } else if (revise_select == 3) {
            // 最大资源需求比例优先
            SortRules.sortTaskByMAXFNRF(taskIdsListOfNode, freqAllocList, currVehicleFreqRemain);
        }

    }


    public static void vehicleArrSortByRules(int[][] vehcileSortArr,
                                             List<Vehicle> vList,
                                             List<Integer> unloadArr,
                                             List<Integer> freqAllocArr) {
        //
        int sort_sel = TaskPolicy.VEHICLE_LIST_SORT_RULE;

        if (sort_sel == 1) {
            // 最大资源剩余量优先
            sortVehicleArrByMAXFRF(vehcileSortArr);
        } else if (sort_sel == 2) {
            // 最小负载比例优先
            sortVehicleArrByMINLRF(vehcileSortArr, vList, unloadArr, freqAllocArr);
        } else if (sort_sel == 3) {
            // 最小请求数优先
            sortVehicleArrByMINRQF(vehcileSortArr, vList, unloadArr, freqAllocArr);
        }
    }


    public static void vehicleSortByRules(List<Vehicle> vList,
                                          List<Integer> unloadArr,
                                          List<Integer> freqAllocArr) {
        int sort_sel = TaskPolicy.VEHICLE_LIST_SORT_RULE;

        if (sort_sel == 1) {
            // 最大资源剩余量优先
            sortVehicleByMAXFRF(vList);
        } else if (sort_sel == 2) {
            // 最小负载比例优先
            sortVehicleByMINLRF(vList, unloadArr, freqAllocArr);
        } else if (sort_sel == 3) {
            // 最小请求数优先
            sortVehicleByMINRQF(vList, unloadArr);
        }
    }

    /**
     * TSR1：最大完成时间优先（Maximum Cost Time First, MAXCTF）
     * 按任务完成时间 降序排列
     * @param nodeGetTaskIDsList
     * @param taskCostTimeList
     */
    public static void sortTaskByMAXCTF(List<Integer> nodeGetTaskIDsList,
                                        List<Double> taskCostTimeList) {
        nodeGetTaskIDsList.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer id1, Integer id2) {
                double ratio1 = taskCostTimeList.get(id1);
                double ratio2 = taskCostTimeList.get(id2);

                if (ratio1 == ratio2) return 0;
                return ratio1 > ratio2 ? -1 : 1;  // 降序
            }
        });
    }

    /**
     * TSR2：最大计算量优先（Maximum Computation Load First, MAXCLF）
     * 按最大计算量优先的顺序将不满足资源卸载的任务修正至其他资源节点（计算量--降序）
     *
     * @param nodeGetTaskIDsList
     * @param taskList
     */
    public static void sortTaskByMAXCLF(List<Integer> nodeGetTaskIDsList,
                                        List<Task> taskList) {
        nodeGetTaskIDsList.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer id1, Integer id2) {
                double c1 = taskList.get(id1).getS() * taskList.get(id1).getC();
                double c2 = taskList.get(id2).getS() * taskList.get(id2).getC();

                if (c1 == c2) return 0;
                return c1 > c2 ? -1 : 1;  // 降序
            }
        });
    }


    /**
     * TSR3：最大资源需求比例优先（Maximum Frequency Ratio First, MAXFNRF）
     * 将占据当前节点计算资源比例更大的任务修改卸载决策至其他目标节点（资源需求比例---降序）
     * @param nodeGetTaskIDsList
     * @param freqAllocList
     * @param currVehicleFreqRemain
     */
    public static void sortTaskByMAXFNRF(List<Integer> nodeGetTaskIDsList,
                                         List<Integer> freqAllocList,
                                         double currVehicleFreqRemain) {
        nodeGetTaskIDsList.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer id1, Integer id2) {
                double c1 = freqAllocList.get(id1);
                double c2 = freqAllocList.get(id2);

                double r1 = c1 / currVehicleFreqRemain;
                double r2 = c2 / currVehicleFreqRemain;

                if (r1 == r2) return 0;
                return r1 > r2 ? -1 : 1;  // 降序
            }
        });
    }


    // RNSS1：最大资源剩余量优先（Maximum Frequency Remain First, MAXFRF）
    public static void sortVehicleArrByMAXFRF(int[][] arr) {
        Arrays.sort(arr, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o2[1] - o1[1]; // 根据第二维 --- 降序
            }
        });
    }

    // RNSS1：最大资源剩余量优先（Maximum Frequency Remain First, MAXFRF）
    public static void sortVehicleArrByMAXFRF(int[][] vehicleSortArr,
                                              List<Vehicle> vList) {
        int vSize = vList.size();

        Arrays.sort(vehicleSortArr, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o2[1] - o1[1];
            }
        });
    }


    // RNSS2：最小负载比例优先（Minimize Load Ratio First, MINLRF）
    public static void sortVehicleArrByMINLRF(int[][] vehicleSortArr,
                                              List<Vehicle> vList,
                                              List<Integer> unloadArr,
                                              List<Integer> freqAllocArr) {
        int vSize = vList.size();
        int[] freqAllocVeh = new int[vSize];
        int len = unloadArr.size();
        for (int i = 0; i < len; i++) {
            int v_id = unloadArr.get(i);
            if (v_id <= 0) continue;
            int freq = freqAllocArr.get(i);
            freqAllocVeh[v_id - 1] += freq;
        }

        // 负载比例
        double[][] loadRatio = new double[vSize][2];
        for (int i = 0; i < vSize; i++) {
            loadRatio[i][0] = vList.get(i).getVehicleID();
            int vehicleFreqRemain = (int) vList.get(i).getFreqRemain();
            // if (vehicleSortArr[i][1] != 0) {
            //     loadRatio[i][1] = freqAllocVeh[i] / vehicleSortArr[i][1];
            // } else {
            //     loadRatio[i][1] = 1.0;
            // }
            if (vehicleFreqRemain != 0) {
                loadRatio[i][1] = freqAllocVeh[i] / vehicleFreqRemain;
            } else {
                loadRatio[i][1] = 1.0;
            }
        }

        Arrays.sort(loadRatio, new Comparator<double[]>() {
            @Override
            public int compare(double[] o1, double[] o2) {
                double r1 = o1[1];
                double r2 = o2[1];
                if (r1 == r2) return 0;
                return r1 > r2 ? 1 : -1;
            }
        });

        int[][] temp = new int[vSize][2];
        for (int i = 0; i < vSize; i++) {
            temp[i][0] = vehicleSortArr[i][0];
            temp[i][1] = vehicleSortArr[i][1];
        }

        for (int i = 0; i < vSize; i++) {
            vehicleSortArr[i][0] = temp[(int) (loadRatio[i][0] - 1)][0];
            vehicleSortArr[i][1] = temp[(int) (loadRatio[i][0] - 1)][1];
        }

    }

    // RNSS3：最小请求数优先（Minimize Request Quantity First, MINRQF）
    public static void sortVehicleArrByMINRQF(int[][] vehicleSortArr,
                                           List<Vehicle> vList,
                                           List<Integer> unloadArr,
                                           List<Integer> freqAllocArr) {
        int vSize = vList.size();

        int len = unloadArr.size();
        // 记录请求数  v_id -- nums
        int[][] requestNums = new int[vSize][2];
        // v_id
        for (int i = 0; i < vSize; i++) {
            requestNums[i][0] = i + 1;
        }
        // 请求数
        for (int i = 0; i < len; i++) {
            int v_id = unloadArr.get(i);
            if (v_id <= 0) continue;
            requestNums[v_id - 1][1] += 1;
        }

        Arrays.sort(requestNums, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[1] - o2[1];
            }
        });

        int[][] temp = new int[vSize][2];
        for (int i = 0; i < vSize; i++) {
            temp[i][0] = vehicleSortArr[i][0];
            temp[i][1] = vehicleSortArr[i][1];
        }

        for (int i = 0; i < vSize; i++) {
            vehicleSortArr[i][0] = temp[(int) (requestNums[i][0] - 1)][0];
            vehicleSortArr[i][1] = temp[(int) (requestNums[i][0] - 1)][1];
        }

    }

    /**
     * RNSS1：最大资源剩余量优先（Maximum Frequency Remain First, MAXFRF）
     * @param vList
     */
    public static void sortVehicleByMAXFRF(List<Vehicle> vList) {
        vList.sort(new Comparator<Vehicle>() {
            @Override
            public int compare(Vehicle o1, Vehicle o2) {
                int r1 = (int) o1.getFreqRemain();
                int r2 = (int) o2.getFreqRemain();

                if (r1 == r2) return 0;
                return r1 > r2 ? -1 : 1; // 降序
            }
        });
    }

    /**
     * RNSS2：最小负载比例优先（Minimize Load Ratio First, MINLRF）
     * @param vList
     */
    public static void sortVehicleByMINLRF(List<Vehicle> vList,
                                           List<Integer> unloadArr,
                                           List<Integer> freqAllocArr) {
        // 车辆已分配出的freq
        HashMap<Integer, Integer> vehicleAllocFreq = new HashMap<>();
        int size = unloadArr.size();
        for (int i = 0; i < size; i++) {
            int unload_node = unloadArr.get(i);
            if (unload_node <= 0) continue;

            vehicleAllocFreq.put(unload_node,
                    vehicleAllocFreq.getOrDefault(unload_node, 0) + freqAllocArr.get(i));
        }

        vList.sort(new Comparator<Vehicle>() {
            @Override
            public int compare(Vehicle o1, Vehicle o2) {
                int size = unloadArr.size();
                int freqRemain_o1 = (int) o1.getFreqRemain();
                int freqRemain_o2 = (int) o2.getFreqRemain();

                int freqAlloc_o1 = vehicleAllocFreq.getOrDefault(o1.getVehicleID(), 0);
                int freqAlloc_o2 = vehicleAllocFreq.getOrDefault(o2.getVehicleID(), 0);

                double r1 = freqAlloc_o1 / freqRemain_o1;
                double r2 = freqAlloc_o2 / freqRemain_o2;

                if (r1 == r2) return 0;
                return r1 > r2 ? 1 : -1; // 降序
            }
        });
    }

    /**
     * RNSS3：最小请求数优先（Minimize Request Quantity First, MINRQF）
     * @param vList
     */
    public static void sortVehicleByMINRQF(List<Vehicle> vList,
                                           List<Integer> unloadArr) {

        // 车辆待卸载的任务数量
        HashMap<Integer, Integer> vehicleGetTaskNum = new HashMap<>();
        int size = unloadArr.size();
        for (int i = 0; i < size; i++) {
            int unload_node = unloadArr.get(i);
            if (unload_node <= 0) continue;

            vehicleGetTaskNum.put(unload_node,
                    vehicleGetTaskNum.getOrDefault(unload_node, 0) + 1);
        }

        vList.sort(new Comparator<Vehicle>() {
            @Override
            public int compare(Vehicle o1, Vehicle o2) {
                int r1 = vehicleGetTaskNum.getOrDefault(o1.getVehicleID(), 0);
                int r2 = vehicleGetTaskNum.getOrDefault(o2.getVehicleID(), 0);

                if (r1 == r2) return 0;
                return r1 > r2 ? 1 : -1; // 升序
            }
        });
    }

    /**
     * 最小 deadline 优先
     *
     * @param nodeGetTaskIDsList
     * @param taskList
     */
    public static void sortMinDeadLineFirstSequence(List<Integer> nodeGetTaskIDsList,
                                                    List<Task> taskList) {
        nodeGetTaskIDsList.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer id1, Integer id2) {
                // double ratio1 = taskList.get(id1).getD() * taskList.get(id1).getFactor();
                // double ratio2 = taskList.get(id2).getD() * taskList.get(id2).getFactor();
                double ratio1 = taskList.get(id1).getD();
                double ratio2 = taskList.get(id2).getD();

                if (ratio1 == ratio2) return 0;
                return ratio1 > ratio2 ? -1 : 1;
            }
        });
    }

    /**
     * 最小 最大deadline 优先
     *
     * @param nodeGetTaskIDsList
     * @param taskList
     */
    public static void sortMinMaxDeadLineFirstSequence(List<Integer> nodeGetTaskIDsList,
                                                       List<Task> taskList) {
        nodeGetTaskIDsList.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer id1, Integer id2) {
                // double ratio1 = taskList.get(id1).getD() * taskList.get(id1).getFactor();
                // double ratio2 = taskList.get(id2).getD() * taskList.get(id2).getFactor();
                double ratio1 = taskList.get(id1).getD();
                double ratio2 = taskList.get(id2).getD();

                if (ratio1 == ratio2) return 0;
                return ratio1 > ratio2 ? -1 : 1;
            }
        });
    }

    /**
     * 最小截止期限优先
     *
     * @param taskList
     */
    public static void sortMinDeadLineFirstSequence(List<Task> taskList) {
        taskList.sort(new Comparator<Task>() {
            @Override
            public int compare(Task o1, Task o2) {
                double ratio1 = o1.getD();
                double ratio2 = o2.getD();

                if (ratio1 == ratio2) return 0;
                return ratio1 > ratio2 ? 1 : -1;
            }
        });
    }

    /**
     * 最小size优先
     *
     * @param taskList
     */
    public static void sortMinDataSizeFirstSequence(List<Task> taskList) {
        taskList.sort(new Comparator<Task>() {
            @Override
            public int compare(Task o1, Task o2) {
                int datasize1 = o1.getS();
                int datasize2 = o2.getS();

                if (datasize1 == datasize2) return 0;
                return datasize1 > datasize2 ? 1 : -1;
            }
        });
    }

    /**
     * 最小 size 优先
     *
     * @param nodeGetTaskIDsList ： 当前node待卸载任务集合
     * @param taskList
     */
    public static void sortMinDataSizeFirstSequence(List<Integer> nodeGetTaskIDsList,
                                                    List<Task> taskList) {
        nodeGetTaskIDsList.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer id1, Integer id2) {
                int datasize1 = taskList.get(id1).getS();
                int datasize2 = taskList.get(id2).getS();

                if (datasize1 == datasize2) return 0;
                return datasize1 > datasize2 ? 1 : -1;  // 升序
            }
        });
    }

    /**
     * 最小计算量(CPU cycles)优先
     *
     * @param taskList
     */
    public static void sortMinComputationFirstSequence(List<Integer> nodeGetTaskIDsList,
                                                       List<Task> taskList) {
        nodeGetTaskIDsList.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer id1, Integer id2) {
                double c1 = taskList.get(id1).getS() * taskList.get(id1).getC();
                double c2 = taskList.get(id2).getS() * taskList.get(id2).getC();

                if (c1 == c2) return 0;
                return c1 > c2 ? 1 : -1;
            }
        });
    }

    public static void sortMinComputationFirstSequence(List<Task> taskList) {
        taskList.sort(new Comparator<Task>() {
            @Override
            public int compare(Task o1, Task o2) {
                double c1 = o1.getS() * o1.getC();
                double c2 = o2.getS() * o2.getC();

                if (c1 == c2) return 0;
                return c1 > c2 ? 1 : -1;
            }
        });
    }

    /**
     * 按任务优先级  p 越小，优先级越高，
     *
     * @param nodeGetTaskIDsList
     * @param taskList
     */
    public static void sortByTaskPrior(List<Integer> nodeGetTaskIDsList,
                                       List<Task> taskList) {
        nodeGetTaskIDsList.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer id1, Integer id2) {
                int p1 = taskList.get(id1).getP();
                int p2 = taskList.get(id2).getP();

                if (p1 == p2) return 0;
                return p1 > p2 ? 1 : -1;  // 从大到小
            }
        });

    }

    /**
     * 用户满意度优先
     */
    public static void sortMaxUssSequence() {

    }

    /**
     * task的优先级排序
     */
    public static void sortByTaskPrior() {

    }

}
