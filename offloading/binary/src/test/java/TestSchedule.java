import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import resource_allocation.PSO_02;
import unload_decision.GA_01;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

@Slf4j
public class TestSchedule {

    public static void main(String[] args) {
        log.debug("TestSchedule ===> main()........");
    }


    /**
    * @Data 2023-12-13
    */
    @Test
    public void testPSO() {
        log.info("TestSchedule ===> 测试PSO算法........");
        PSO_02 p = new PSO_02();
        int numParticles = 30; // 粒子数量
        int numDimensions = 10; // 问题的维度，根据实际问题设置
        int numIterations = 100; // 迭代次数
        double inertiaWeight = 0.5; // 惯性权重
        double cognitiveWeight = 1.5; // 个体认知权重
        double socialWeight = 1.5; // 群体社会权重

        List<Integer> resArr = new ArrayList<>();
        for(int i = 0; i < numDimensions; i++) {
            resArr.add(new Random().nextInt(100) + 1);
        }

        p.initParticles(numParticles, numDimensions, resArr);

        p.optimizeParticles(numIterations, inertiaWeight, cognitiveWeight, socialWeight, 1, 100, -1, -1);

    }

    /**
    * @Data 2023-12-13
    */
    @Test
    public void testGA() {

        log.info("TestSchedule ===> 测试 GA algorithm .......");

        int populationSize = 20; // 种群大小
        double crossoverRate = 0.8; // 交叉概率
        double mutationRate = 0.01; // 变异概率
        int geneLength = 10; // 染色体长度(维度数)
        int numGenerations = 50; // 迭代代数

        List<Integer> unloadDecisionArr = new ArrayList<>(); // 个数应该为 geneLength
        for (int i = 0; i < geneLength; i++) {
            unloadDecisionArr.add(new Random().nextInt(10) + 1);
        }

        GA_01 ga = new GA_01(populationSize, crossoverRate, mutationRate, geneLength);

        ga.initPopulation(unloadDecisionArr);

        ga.optimizeChromosomes(numGenerations);

        log.info("===================================================");
        log.info(Arrays.toString(ga.getCurrBestChromosome().getGene()));

    }


    /**
    * @Data 2023-12-13
     *
     * 1、读入数据 任务集  资源集
     * 2、编号、聚类 ==> 初始资源分配
     * 3、GA ==> 卸载决策
     * 4、贪婪修正
     * 5、PSO  ==> 资源分配
     * 6、贪婪修正
    */
    @Test
    public void testSechedule() {
        // 联合测试
    }


}
