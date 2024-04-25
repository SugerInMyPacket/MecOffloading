package ga;
import java.util.ArrayList;
import java.util.List;

public class Population {
    private List<Individual> individuals;
    private int populationSize;
    private double mutationRate;
    private double minValue;
    private double maxValue;

    public Population(int populationSize, double mutationRate, double minValue, double maxValue) {
        this.populationSize = populationSize;
        this.mutationRate = mutationRate;
        this.minValue = minValue;
        this.maxValue = maxValue;
        individuals = new ArrayList<>();
    }

    public void initializePopulation() {
        for (int i = 0; i < populationSize; i++) {
            individuals.add(new Individual(generateRandomGenes()));
        }
    }

    private double[] generateRandomGenes() {
        double[] genes = new double[2]; // Assuming 2 genes for this example
        for (int i = 0; i < genes.length; i++) {
            genes[i] = minValue + (maxValue - minValue) * Math.random();
        }
        return genes;
    }

    public void evolve() {
        // Evolve the population
    }

    public Individual getBestIndividual() {
        // Get the best individual in the population
        return individuals.get(0);
    }
}
