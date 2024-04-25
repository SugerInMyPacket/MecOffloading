package utils;

import java.util.Random;

public class NumUtil {
    /**
     * 获取随机值          min  <=  result  <=  max
     *
     * @return
     */
    public static Double random(Double min, Double max) {
        Random rand = new Random();
        double result = 0;
        for (int i = 0; i < 10; i++) {
            result = min + (rand.nextDouble() * (max - min));
            result = (double) Math.round(result * 100) / 100.0;
        }
        return result;
    }

    /**
     *  随机数 服从N(0,1）
     *
     *  若随机变量X服从一个数学期望为μ、方差为σ^2的正态分布，记为X~N(μ，σ^2)。
     *
     * 其概率密度函数为正态分布的期望值μ决定了其位置，其标准差σ决定了分布的幅度。当μ = 0,σ = 1时的正态分布是标准正态分布。
     */
    public static Double random(){
        Random r = new Random();
        return r.nextGaussian();
    }

}
