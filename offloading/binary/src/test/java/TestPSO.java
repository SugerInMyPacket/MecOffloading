import org.junit.Test;
import resource_allocation.PSO_02;
import utils.FormatData;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

public class TestPSO {

    public static void main(String[] args) {
        System.out.println("main()......");
    }

    @Test
    public void testFormatData() {
        System.out.println(FormatData.getEffectiveValue4Digit(3.1452864707, 3));
    }

    /**
    * @Data 2023-12-13
    */
    @Test
    public void testPso02() {
        PSO_02 p = new PSO_02();
        int numParticles = 30; // 粒子数量
        int numDimensions = 5; // 问题的维度，根据实际问题设置
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
    * @Data 2024-01-11
    */
    @Test
    public void testSort() {
        List<Integer> list = new ArrayList<>();
        list.add(1);
        list.add(4);
        list.add(2);
        list.add(8);
        list.add(67);

        list.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                if(o1 == o2) return 0;
                return o1 > o2 ? 1 : -1;  // 升序
            }
        });
        System.out.println(list);
    }
}
