package ga;

public class Individual {
    private double[] genes;

    public Individual(double[] genes) {
        this.genes = genes;
    }

    public double[] getGenes() {
        return genes;
    }

    public double fitness1() {
        double x = genes[0];
        double y = genes[1];
        return Math.sin(x) * Math.cos(y);
    }

    public double fitness2() {
        double x = genes[0];
        double y = genes[1];
        return Math.sin(y) * Math.cos(x);
    }

    @Override
    public String toString() {
        return "Genes: " + java.util.Arrays.toString(genes) + ", Fitness: (" + fitness1() + ", " + fitness2() + ")";
    }
}

