package unload_decision;

public class Chromosome {
    int[] gene;
    double fitness;

    public Chromosome() {}

    public Chromosome(int geneLength) {
        gene = new int[geneLength];
        for (int i = 0; i < geneLength; i++) {
            this.gene[i] = 0;
        }
    }

    public int[] getGene() {
        return gene;
    }
}
