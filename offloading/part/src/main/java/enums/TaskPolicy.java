package enums;

public class TaskPolicy {

    // KEYNOTE: 节点待卸载任务的排序方式
    // 0、预处理 GFC
    // 1、优先级（降序）  2、size（升序）   3、截止时间（升序） 4、最小 max_deadline 优先
    // 1、最短容忍截止期限优先（升序）  2、最小执行时间优先（升序）   3、最小计算量优先级之积优先（升序）
    public static final int TASK_LIST_SORT_RULE_LOCAL = 2;
    public static final int TASK_LIST_SORT_RULE_REMOTE = 2;


    // 任务 聚类
    // class_0: 计算量 【小】, 截止时间 【短】  --- black
    public static final int TASK_CLUSTER_CLASS_0 = 0;
    public static final int TASK_FREQ_INIT_CLASS_0 = 200;
    // class_1: 计算量 【大】, 截止时间 【短】  --- purple
    public static final int TASK_CLUSTER_CLASS_1 = 1;
    public static final int TASK_FREQ_INIT_CLASS_1 = 400;
    // class_2: 计算量 【小】, 截止时间 【长】  --- yellow
    public static final int TASK_CLUSTER_CLASS_2 = 2;
    public static final int TASK_FREQ_INIT_CLASS_2 = 100;
    // class_3: 计算量 【大】, 截止时间 【长】  --- cyan
    public static final int TASK_CLUSTER_CLASS_3 = 3;
    public static final int TASK_FREQ_INIT_CLASS_3 = 200;

    public static final int TASK_FREQ_INIT_CLASS_DEFAULT = 200;

}
