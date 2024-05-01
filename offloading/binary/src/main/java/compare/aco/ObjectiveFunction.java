package compare.aco;

public class ObjectiveFunction {
    static double[] evaluate(double[] solution) {
        // 计算目标函数
        double[] objectives = new double[2];
        double sum1 = 0, sum2 = 0;
        for (int i = 0; i < solution.length / 2; i++) {
            sum1 += solution[i];
            sum2 += solution[solution.length / 2 + i];
        }
        objectives[0] = sum1;
        objectives[1] = sum2;
        return objectives;
    }
}
