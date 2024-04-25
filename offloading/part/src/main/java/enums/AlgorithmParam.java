package enums;

public class AlgorithmParam {

    // 迭代轮数
    public static final int NUM_ROUND_TIMES = 5;

    // 优化代数
    public static final int NUM_ITERATIONS = 50;


    // GA 相关参数
    // 染色体数量
    public static final int NUM_CHROMOSOMES = 100;
    // GA粒子卸载决策边界
    public static final int MAX_NUM_GA = 10;
    public static final int MIN_NUM_GA = -1;

    // 差分进化算法相关参数
    // 个体数量
    public static final int NUM_POPULATIONS = 100;
    public static final double CROSSOVER_RATE_DE = 0.5;
    public static final double MUTATION_FACTOR_DE = 0.5;


    // 模拟退火算法相关参数
    // 初始温度
    public static final Double INIT_TEMPERATURE = 100.0;
    // 最大迭代次数
    public static final Integer ITERATION_TIMES_SA = 2000;
    // 每个温度下的迭代次数
    public static final Integer CURRENT_TEMP_ITERATION_TIMES = 50;
    // 温度衰减系数
    public static final Double TEMP_DECREASE_RATE = 0.95;
    // x的上、下界
    public static final Double MIN_X = 0.0;
    public static final Double MAX_X = 1.0;


    // PSO 相关参数
    public static final int NUM_PARTICLES = 100;
    // 惯性参数 w
    public static final double INERTIA_WEIGHT = 0.5;
    public static final double MAX_INERTIA_WEIGHT = 1.5;
    public static final double MIN_INERTIA_WEIGHT = 0.5;
    // 个体学习因子 ɗ_1
    public static final double COGNITIVE_WEIGHT = 1.5;
    public static final double MAX_COGNITIVE_WEIGHT = 3.2;
    public static final double MIN_COGNITIVE_WEIGHT = 0.8;
    // 社会学习因子 ɗ_1
    public static final double SOCIAL_WEIGHT = 1.5;
    public static final double MAX_SOCIAL_WEIGHT = 3.4;
    public static final double MIN_SOCIAL_WEIGHT = 1.2;

    // PSO粒子速度边界
    public static final double MAX_VEL_PARTICLE = 20;
    public static final double MIN_VEL_PARTICLE = 100;
    // PSO粒子位置边界
    public static final double MIN_POS_PARTICLE = 100;
    public static final double MAX_POS_PARTICLE = 600;


    // uss 和 energy 在粒子替换时的占比
    public static final double FITNESS_RATIO = 0.6;

}
