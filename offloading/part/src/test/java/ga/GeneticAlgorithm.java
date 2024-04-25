package ga;


public class GeneticAlgorithm {
    public static void main(String[] args) {
        int populationSize = 100;
        double mutationRate = 0.01;
        int maxGenerations = 1000;
        double minValue = -10.0;
        double maxValue = 10.0;

        Population population = new Population(populationSize, mutationRate, minValue, maxValue);
        population.initializePopulation();

        for (int i = 0; i < maxGenerations; i++) {
            population.evolve();
            System.out.println("Generation " + i + ": Best Fitness = " + population.getBestIndividual().fitness1() + ", " + population.getBestIndividual().fitness2());
        }

        System.out.println("Best solution found: " + population.getBestIndividual());
    }
}

