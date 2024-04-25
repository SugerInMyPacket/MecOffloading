package unload_decision;

import java.util.List;
import java.util.Random;

public class Chromosome4 {
    int[] gene;
    double[] fitness;

    public Chromosome4() {}

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

    public Chromosome4(int geneLength,
                       int bound,
                       List<Integer> currFreqAllocArr) {
        Random random = new Random();

        gene = new int[geneLength];

        // 双目标优化
        fitness = new double[2];

        for (int i = 0; i < geneLength; i++) {
            this.gene[i] = random.nextInt(bound);
        }

        fitness = ObjFuncGA2.evaluate(gene, currFreqAllocArr);

    }

    public Chromosome4(int geneLength,
                       int bound,
                       List<Integer> currUnloadArr,
                       List<Integer> currFreqAllocArr) {
        Random random = new Random();

        gene = new int[geneLength];

        // 双目标优化
        fitness = new double[2];

        for (int i = 0; i < geneLength; i++) {
            this.gene[i] = currUnloadArr.get(i);
        }

        // 适应度评估
        fitness = ObjFuncGA2.evaluate(gene, currFreqAllocArr);

    }


}
