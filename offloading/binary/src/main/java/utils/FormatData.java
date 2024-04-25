package utils;

import java.util.ArrayList;
import java.util.List;

public class FormatData {

    public static double getEffectiveValue4Digit(double oldValue, int digit) {
        double newValue = 0.0;
        // TODO：获取 oldValue 的有效digit位
        if(digit == 0) {
            newValue = (int)oldValue;
            return newValue;
        }
        double multiplierNum = Math.pow(10, digit);
        // newValue = (double)((int)(oldValue * multiplierNum)) / multiplierNum;
        newValue = (int)(oldValue * multiplierNum) / multiplierNum;
        return newValue;
    }

    public static List<Double> getEffectiveValueList4Digit(List<Double> oldValList, int digit) {
        List<Double> newValList = new ArrayList<>();
        int size = oldValList.size();

        for (int i = 0; i < size; i++) {
            double oldVal = oldValList.get(i);
            double newVal = getEffectiveValue4Digit(oldVal, digit);
            newValList.add(newVal);
        }
        return newValList;
    }

    public static double[] getEffectiveValue4Digit(double[] oldValue, int digit) {
        int len = oldValue.length;
        double[] newValue = new double[len];
        // TODO：获取 oldValue 的有效digit位

        if(digit == 0) {
            for (int i = 0; i < len; i++) {
                int intNum = (int) oldValue[i];
                newValue[i] = intNum;
            }
            return newValue;
        }

        double changeNum = 1.0 * Math.pow(10, digit);
        for (int i = 0; i < len; i++) {
            newValue[i] = (int)(oldValue[i] * changeNum) / changeNum;
        }
        return newValue;
    }

    // 四舍五入
    public static int halfAdjustValue(double oldVal) {
        int result = 0;
        int temp = (int)(oldVal * 10 % 10);
        if(temp >= 5) {
            result = (int)oldVal + 1;
        } else {
            result = (int)oldVal;
        }
        return result;
    }
}
