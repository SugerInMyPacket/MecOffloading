package resource_allocation;

public interface BasePSO {

    /*
    * 初始化粒子表示
    * */
    void initParticles();

    /**
     * 计算适应度
     */
    double evaluateFitness();

    /**
     * 粒子迭代优化过程
     */
    void optimizeParticles();

}
