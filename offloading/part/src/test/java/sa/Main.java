package sa;

// Main.java
public class Main {
    public static void main(String[] args) {
        // Define your objective function
        ObjectiveFunction objectiveFunction = new ObjectiveFunction() {
            @Override
            public double evaluate(double[] solution) {
                // Implement your objective function here
                // Return the evaluation of the solution
                return 0.0; // Placeholder, replace with actual evaluation
            }
        };

        // Create SimulatedAnnealing instance
        SimulatedAnnealing sa = new SimulatedAnnealing(objectiveFunction);

        // Set initial solution, temperature, cooling rate, and maximum iterations
        double[] initialSolution = new double[]{0.0, 0.0}; // Initial solution
        double initialTemperature = 1000.0; // Initial temperature
        double coolingRate = 0.95; // Cooling rate
        int maxIterations = 10000; // Maximum iterations

        // Solve the problem
        double[] solution = sa.solve(initialSolution, initialTemperature, coolingRate, maxIterations);

        // Print the result
        System.out.println("Optimal Solution:");
        for (double value : solution) {
            System.out.println(value);
        }
    }
}
