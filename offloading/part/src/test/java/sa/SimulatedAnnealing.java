package sa;// SimulatedAnnealing.java
import java.util.Random;

public class SimulatedAnnealing {
    private final ObjectiveFunction objectiveFunction;
    private final Random random;

    public SimulatedAnnealing(ObjectiveFunction objectiveFunction) {
        this.objectiveFunction = objectiveFunction;
        this.random = new Random();
    }

    public double[] solve(double[] initialSolution,
                          double initialTemperature,
                          double coolingRate,
                          int maxIterations) {
        double[] currentSolution = initialSolution;
        double[] bestSolution = initialSolution;
        double currentEnergy = objectiveFunction.evaluate(initialSolution);
        double bestEnergy = currentEnergy;

        double temperature = initialTemperature;

        for (int i = 0; i < maxIterations; i++) {
            double[] newSolution = generateNeighbor(currentSolution);
            double newEnergy = objectiveFunction.evaluate(newSolution);


            if (acceptanceProbability(currentEnergy, newEnergy, temperature) > random.nextDouble()) {
                currentSolution = newSolution;
                currentEnergy = newEnergy;
            }

            if (currentEnergy < bestEnergy) {
                bestSolution = currentSolution;
                bestEnergy = currentEnergy;
            }

            temperature *= coolingRate;
        }

        return bestSolution;
    }

    private double[] generateNeighbor(double[] solution) {
        // Implement neighbor generation logic here
        // For example, you can generate a neighbor by perturbing the solution
        // randomly within certain bounds
        return solution;
    }

    private double acceptanceProbability(double energy, double newEnergy, double temperature) {
        if (newEnergy < energy) {
            return 1.0;
        }
        return Math.exp((energy - newEnergy) / temperature);
    }
}
