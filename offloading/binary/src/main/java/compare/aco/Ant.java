package compare.aco;

public class Ant {
    double[] solution;
    double[] pheromone;
    double[] objectives;

    Ant(int n) {
        solution = new double[2 * n];
        pheromone = new double[2 * n];
        objectives = new double[2];
    }
}
