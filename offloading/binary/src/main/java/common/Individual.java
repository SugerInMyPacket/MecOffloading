package common;

import lombok.extern.slf4j.Slf4j;

import java.util.List;

/**
 * 调度粒子个体
 * gene ===>
 *
 */


@Slf4j
public class Individual {

    int popSize;

    // GA 相关信息 & 适应度计算函数
    int[] gene;
    double[] fitness;

    public static double getFitnessGA(Individual idl) {

        return 0;
    }


    // PSO 相关信息 & 适应度计算函数
    // 坐标
    double[] position;
    // 速度
    double[] velocity;
    // 个体最佳位置
    double[] pBestPosition;
    // 最佳适应度
    double[] pBestFitness;

    public static double getFitnessPSO(Individual idl) {

        return 0;
    }

    public Individual() {}


    public void initIndividualUnload(int[] unloadArr, int[] freqAllocArr) {
        for (int i = 0; i < position.length; i++) {
            position[i] = freqAllocArr[i];
        }


    }

    public void initIndividualFreq(int[] unloadArr, int[] freqAllocArr) {
        for (int i = 0; i < gene.length; i++) {
            gene[i] = unloadArr[i];
        }
    }


}
