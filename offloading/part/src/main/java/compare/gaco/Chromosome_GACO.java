package compare.gaco;

import utils.ArrayUtils;
import utils.NumUtil;

import java.util.List;
import java.util.Random;

public class Chromosome_GACO {

    double[] geneUnload;
    double[] geneRatio;
    double[] geneFreqLocal;
    double[] geneFreqRemote;
    double[] fitness;

    double[] gene;

    public Chromosome_GACO() {}


    public Chromosome_GACO(int geneLength, double bound1, double[][] bound3, double[][] bound4) {
        Random random = new Random();
        gene = new double[geneLength * 4];
        geneUnload = new double[geneLength];
        geneRatio = new double[geneLength];
        geneFreqLocal = new double[geneLength];
        geneFreqRemote = new double[geneLength];
        fitness = new double[2];

        for (int i = 0; i < geneLength; i++) {
            this.geneUnload[i] = -1 + random.nextDouble() * (bound1 + 1);
            this.geneRatio[i] = NumUtil.random(0.0, 1.0);
            this.geneFreqLocal[i] =  100 + random.nextDouble() * 400;
            this.geneFreqRemote[i] = 100 + random.nextDouble() * 400;
            // this.geneFreqLocal[i] =  bound3[i][0] + random.nextDouble() * (bound3[i][1] - bound3[i][0]);
            // this.geneFreqRemote[i] = bound4[i][0] + random.nextDouble() * (bound4[i][1] - bound4[i][0]);
        }
        gene = ArrayUtils.connectArrays(geneUnload, geneRatio, geneFreqLocal, geneFreqRemote);
    }

    public double[] getGene() {
        // gene = ArrayUtils.connectArrays(geneUnload, geneRatio, geneFreqLocal, geneFreqRemote);
        return gene;
    }

}
