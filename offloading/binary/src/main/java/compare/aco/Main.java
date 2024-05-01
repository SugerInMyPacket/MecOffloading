package compare.aco;

public class Main {
    public static void main(String[] args) {
        final int N = 10; // 定义个体长度
        final int ITERATIONS = 100; // 迭代次数
        final int ANTS = 10; // 蚂蚁数量
        final int K = 10; // 范围参数

        double[] bestSolution = AntColonyAlgorithm.solve(N, ITERATIONS, ANTS, K);
        System.out.println("Best Solution:");
        for (double value : bestSolution) {
            System.out.print(value + " ");
        }
        System.out.println();
        System.out.println("Best Objectives:");
        double[] objectives = ObjectiveFunction.evaluate(bestSolution);
        System.out.println("Objective 1: " + objectives[0]);
        System.out.println("Objective 2: " + objectives[1]);
    }
}
