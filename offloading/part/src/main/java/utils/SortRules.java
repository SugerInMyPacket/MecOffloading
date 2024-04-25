package utils;

import entity.Task;
import enums.Constants;
import enums.TaskPolicy;

import java.util.Comparator;
import java.util.List;

public class SortRules {
    /**
     * Sort Rule
     * @param taskIdsListOfNode : 当前节点待卸载的任务id-list
     * @param taskList
     */
    public static void sortListByRules(List<Integer> taskIdsListOfNode,
                                       List<Task> taskList) {

        // 任务排序规则
        int revise_select = TaskPolicy.TASK_LIST_SORT_RULE_REMOTE;

        if (revise_select == 1) {
            // 优先级排序（降序）
            SortRules.sortByTaskPrior(taskIdsListOfNode, taskList);
        } else if (revise_select == 2) {
            // 计算量排序（升序）
            SortRules.sortMinComputationFirstSequence(taskIdsListOfNode, taskList);
        } else if (revise_select == 3) {
            // deadline 排序（升序）
            SortRules.sortMinDeadLineFirstSequence(taskIdsListOfNode, taskList);
        } else if (revise_select == 4) {
            SortRules.sortMinMaxDeadLineFirstSequence(taskIdsListOfNode, taskList);
        }
    }

    /**
     * Sort Rule
     * @param taskIdsListOfNode : 当前节点待卸载的任务id-list
     * @param taskList
     */
    public static void sortLocalTaskListByRules(List<Integer> taskIdsListOfNode,
                                       List<Task> taskList,
                                       List<Integer> freqAllocArr) {

        // 本地子任务 -- 排序规则
        int revise_select = TaskPolicy.TASK_LIST_SORT_RULE_LOCAL;

        if (revise_select == 1) {
            // 最短容忍截止期限优先（升序）
            SortRules.sortLocalTaskBySTDF(taskIdsListOfNode, taskList);
        } else if (revise_select == 2) {
            // 最小执行时间优先（升序）
            SortRules.sortLocalTaskByMINETF(taskIdsListOfNode, taskList, freqAllocArr);
        } else if (revise_select == 3) {
            // 最小计算量优先级之积优先（升序）
            SortRules.sortLocalTaskByMINPCPF(taskIdsListOfNode, taskList);
        }
    }

    // LTSR1：最短容忍截止期限优先（Shortest Tolerance Deadline First, STDF）
    public static void sortLocalTaskBySTDF(List<Integer> nodeGetTaskIDsList,
                                           List<Task> taskList) {
        nodeGetTaskIDsList.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer id1, Integer id2) {
                double ratio1 = taskList.get(id1).getD() * taskList.get(id1).getFactor();
                double ratio2 = taskList.get(id2).getD() * taskList.get(id2).getFactor();

                if (ratio1 == ratio2) return 0;
                return ratio1 > ratio2 ? 1 : -1;  // 【升序】
            }
        });
    }

    // LTSR2：最小执行时间优先（Minimize Execution Time First， MINETF）
    public static void sortLocalTaskByMINETF(List<Integer> nodeGetTaskIDsList,
                                        List<Task> taskList,
                                        List<Integer> freqAllocArr) {
        nodeGetTaskIDsList.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer id1, Integer id2) {
                double CL1 = taskList.get(id1).getS() * taskList.get(id1).getC();
                double CL2 = taskList.get(id1).getS() * taskList.get(id1).getC();

                double ratio1 = CL1 / freqAllocArr.get(id1);
                double ratio2 = CL2 / freqAllocArr.get(id2);

                if (ratio1 == ratio2) return 0;
                return ratio1 > ratio2 ? 1 : -1;  // 【升序】
            }
        });
    }

    // LTSR3：最小计算量优先级之积优先（Minimize Product of Computation and Priority First, MINPCPF）：
    public static void sortLocalTaskByMINPCPF(List<Integer> nodeGetTaskIDsList,
                                         List<Task> taskList) {
        nodeGetTaskIDsList.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer id1, Integer id2) {
                double ratio1 = taskList.get(id1).getS() * taskList.get(id1).getC() * taskList.get(id1).getP();
                double ratio2 = taskList.get(id2).getS() * taskList.get(id2).getC() * taskList.get(id2).getP();

                if (ratio1 == ratio2) return 0;
                return ratio1 > ratio2 ? 1 : -1;  // 【升序】
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
     * 最小 max_deadline 优先
     * @param nodeGetTaskIDsList
     * @param taskList
     */
    public static void sortMinMaxDeadLineFirstSequence(List<Integer> nodeGetTaskIDsList,
                                                       List<Task> taskList) {
        nodeGetTaskIDsList.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer id1, Integer id2) {
                double ratio1 = taskList.get(id1).getD() * taskList.get(id1).getFactor();
                double ratio2 = taskList.get(id2).getD() * taskList.get(id2).getFactor();

                if (ratio1 == ratio2) return 0;
                return ratio1 > ratio2 ? 1 : -1;  // 【升序】
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
                return c1 > c2 ? 1 : -1;  // 【升序】
            }
        });
    }


    /**
     * 按任务优先级  ---> p 越小，优先级越高
     * 故应【降序排列】，是p大的在remote卸载
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
                return p1 > p2 ? -1 : 1;  // 【降序】从大到小
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