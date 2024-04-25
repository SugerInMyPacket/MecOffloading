package enums;

public class AlgorithmParam {

    // 迭代轮数
    public static final int NUM_ROUND_TIMES = 10;

    // 优化代数
    public static final int NUM_ITERATIONS = 50;

    // 粒子数量
    public static final int NUM_PARTICLES = 100;
    public static final int NUM_CHROMOSOMES = 100;
    public static final int NUM_INDIVIDUALS = 100;

    // PSO粒子速度边界
    public static final double MAX_VEL_PARTICLE = 10;
    public static final double MIN_VEL_PARTICLE = 100;

    // PSO粒子位置边界
    public static final double MAX_POS_PARTICLE = 500;
    public static final double MIN_POS_PARTICLE = 100;

    // GA粒子卸载决策
    public static final int MAX_NUM_GA = 10;
    public static final int MIN_NUM_GA = -1;


    // uss 和 energy 在粒子替换时的占比
    public static final double FITNESS_RATIO = 0.6;

}
