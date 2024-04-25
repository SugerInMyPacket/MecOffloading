package unload_decision;

import java.util.List;
import java.util.Random;

public class Chromosome2 {
    int[] gene;
    double[] fitness;

    public Chromosome2() {}

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

    public Chromosome2(int geneLength, int bound, List<Integer> currFreqAllocArr) {
        Random random = new Random();

        gene = new int[geneLength];

        // 双目标优化
        fitness = new double[2];

        for (int i = 0; i < geneLength; i++) {
            this.gene[i] = random.nextInt(bound);
        }

        fitness = ObjFunctionGA.evaluate(currFreqAllocArr, gene);

    }



}
