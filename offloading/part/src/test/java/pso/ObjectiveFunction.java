package pso;

public class ObjectiveFunction {
    // 根据你的目标函数进行修改
    static double[] evaluate(double[] x) {
        double[] result = new double[2];

        result[0] = x[0] * x[0]; // 目标函数1

        result[1] = (x[0] - 2) * (x[0] - 2) + x[1] * x[1]; // 目标函数2

        return result;
    }
}
