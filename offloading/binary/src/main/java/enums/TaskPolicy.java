package enums;

public class TaskPolicy {

    // KEYNOTE: 节点待卸载任务的排序方式
    // 0、预处理 GFC
    // 1、最大完成时间优先  2、最大计算量优先   3、最大资源需求比例优先
    public static final int TASK_LIST_SORT_RULE = 3;

    // 车辆排序规则
    // 1、最大资源剩余量优先  2、最小负载比例优先  3、最小请求数优先
    public static final int VEHICLE_LIST_SORT_RULE = 1;

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
