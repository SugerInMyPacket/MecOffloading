package utils;

public class ArrayUtils {

    // 拼接多个数组
    public static double[] connectArrays(double[] ... arrays) {
        int len = arrays.length;
        int nums = arrays[0].length;

        double[] result = new double[nums * len];

        for (int i = 0; i < len; i++) {
            for (int j = 0; j < nums; j++) {
                result[i * nums + j] = arrays[i][j];
            }
        }

        return result;
    }
}
