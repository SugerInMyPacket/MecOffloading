package unload_decision;

import java.util.List;
import java.util.Random;

public class Chromosome {
    int[] gene;
    double[] fitness;

    public Chromosome() {}

    /*
    public Chromosome2(int geneLength) {
        gene = new int[geneLength];
        for (int i = 0; i < geneLength; i++) {
            this.gene[i] = 0;
        }

        // 双目标优化
        fitness = new double[2];
    }
     */

    public Chromosome(int geneLength,
                      int bound,
                      List<Double> unloadRatioArr,
                      List<Integer> currFreqAllocArrLocal,
                      List<Integer> currFreqAllocArrRemote) {
        Random random = new Random();

        gene = new int[geneLength];

        // 双目标优化
        fitness = new double[2];

        for (int i = 0; i < geneLength; i++) {
            this.gene[i] = random.nextInt(bound);
        }

        fitness = ObjFuncGA.evaluate(gene, unloadRatioArr,
                currFreqAllocArrLocal, currFreqAllocArrRemote);

    }



}
