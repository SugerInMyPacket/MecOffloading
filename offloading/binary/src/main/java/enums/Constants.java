package enums;

public class Constants {
    public static final int GFC_ON = 1;
    // 满意度计算阈值
    public static final int USS_THRESHOLD = 11;
    public static final double OBJ_ALPHA = 0.5;

    // 数据
    public static final String VehicleDB = "vehicle3_50";
    public static final String TaskDB = "task_500_v50";

    public static final double DATA_SIZE_MULTI_INCREASE = 0.5;

    // 车辆数
    public static final int VEHICLE_NUMS = 50;

    // 剩余资源量
    public static final int FREQ_REMAIN_RSU = 5000;
    public static final int FREQ_REMAIN_Cloud = 300000;

    // 信道带宽
    public static final int WIDTH_CHANNEL_VEHICLE = 2;
    public static final int WIDTH_CHANNEL_RSU = 20;
    public static final int WIDTH_CHANNEL_CLOUD = 5;
    // public static final int WIDTH_CHANNEL_VEHICLE = 20;
    // public static final int WIDTH_CHANNEL_RSU = 200;
    // public static final int WIDTH_CHANNEL_CLOUD = 400;

    // 功率系数 （计算任务执行能耗）
    public static final double COEFFICIENT_POWER_VEHICLE = 1;
    public static final double COEFFICIENT_POWER_RSU = 1.1;
    public static final double COEFFICIENT_POWER_CLOUD = 1.3;

    // 传输功率 µ
    public static final long µ = 1;
    public static final Long POWER_TRANS_VEHICLE = 10000l;
    // public static final Long POWER_TRANS_RSU = 200000l;
    public static final int TRANS_R2C_MULTI_HOP = 10;
    public static final Long POWER_TRANS_RSU = 20000l;
    public static final Long POWER_TRANS_CLOUD = 600000l;

    // 噪声功率
    public static final Long POWER_NOISE = 1l;
    public static final int A0 = 1;

    // 信道增益
    public static final Long GAIN_CHANNEL_R2C = 1l;
    public static final Long GAIN_CHANNEL_C2R = 1L;
    public static final Long GAIN_CHANNEL_V2R = 1L;
    public static final Long GAIN_CHANNEL_R2V = 1L;
    public static final Long GAIN_CHANNEL_V2V = 1L;

    // 时间槽
    public static final double TIME_SLOT_DURATION = 1.0;

    // 粒子数量
    public static final int NUM_PARTICLES = 100;
    public static final int NUM_CHROMOSOMES = 100;
    public static final int NUM_INDIVIDUALS = 100;

    // 优化代数
    public static final int NUM_ITERATIONS = 50;
    // 迭代轮数
    public static final int NUM_ROUND_TIMES = 10;

    // PSO粒子速度边界
    public static final double MAX_VEL_PARTICLE = 100;
    public static final double MIN_VEL_PARTICLE = 300;

    // PSO粒子位置边界
    public static final double MAX_POS_PARTICLE = 500;
    public static final double MIN_POS_PARTICLE = 100;

    // GA粒子卸载决策
    public static final int MAX_NUM_GA = 10;
    public static final int MIN_NUM_GA = -1;


    // uss 和 energy 在粒子替换时的占比
    public static final double FITNESS_RATIO = 0.7;


}
