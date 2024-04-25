package pso;

public class Main {
    public static void main(String[] args) {
        int numParticles = 20;
        int numDimensions = 2;
        double[][] bounds = {{-10, 10}, {-10, 10}};
        double inertiaWeight = 0.7;
        double cognitiveWeight = 1.5;
        double socialWeight = 1.5;
        int maxIterations = 100;
        
        PSO pso = new PSO(numParticles, numDimensions, bounds, inertiaWeight, cognitiveWeight, socialWeight);
        pso.optimize(maxIterations);
    }
}
