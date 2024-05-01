package compare.gaco;

import config.InitFrame;
import config.RevisePolicy;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import utils.FormatData;
import utils.Formula;

import java.util.ArrayList;
import java.util.List;

public class ObjFunc_GACO {

    public static double[] evaluate(double[] currUnloadArr,
                                    double[] currUnloadRatioArr,
                                    double[] currFreqLocalArr,
                                    double[] currFreqRemoteArr) {

        double[] result = new double[2];

        int dim = currUnloadArr.length;

        // 当前粒子的卸载决策
        List<Integer> currParticleUnloadArr = new ArrayList<>();
        List<Double> currParticleUnloadRatioArr = new ArrayList<>();
        for (int i = 0; i < dim; i++) {
            currParticleUnloadArr.add((int) currUnloadArr[i]);
            currParticleUnloadRatioArr.add(currUnloadRatioArr[i]);
        }

        // 当前粒子的资源分配方案
        List<Integer> currParticleFreqLocalArr = new ArrayList<>();
        List<Integer> currParticleFreqRemoteArr = new ArrayList<>();
        for (int i = 0; i < dim; i++) {
            currParticleFreqLocalArr.add((int) currFreqLocalArr[i]);
            currParticleFreqRemoteArr.add((int) currFreqRemoteArr[i]);
        }

        // 资源list
        List<Task> taskList = InitFrame.getTaskList();
        List<Vehicle> vehicleList = InitFrame.getVehicleList();
        RoadsideUnit rsu = InitFrame.getRSU();

        // 计算uss和energy
        List<Double> currUssList = new ArrayList<>();
        List<Double> currEnergyList = new ArrayList<>();

        // 修正卸载
        RevisePolicy.reviseUnloadArrRemote(taskList, vehicleList, rsu,
                currParticleUnloadArr, currParticleFreqRemoteArr);

        // 修正后的值返回
        for (int i = 0; i < dim; i++) {
            currUnloadArr[i] = currParticleUnloadArr.get(i);
            currFreqRemoteArr[i] = currParticleFreqRemoteArr.get(i);
        }

        // 重新计算uss和energy
        currUssList
                = Formula.getUSS4TaskList(taskList, currParticleUnloadArr, currParticleUnloadRatioArr,
                currParticleFreqLocalArr, currParticleFreqRemoteArr);
        currEnergyList
                = Formula.getEnergy4TaskList(taskList, currParticleUnloadArr, currParticleUnloadRatioArr,
                currParticleFreqLocalArr, currParticleFreqRemoteArr);

        // USS 总和 & energy 总和
        double ussTotal = 0.0;
        double energyTotal = 0.0;
        for (int i = 0; i < dim; i++) {
            ussTotal += currUssList.get(i);
            energyTotal += currEnergyList.get(i);
        }

        result[0] = ussTotal / (double) dim;
        result[1] = -energyTotal / (double) dim / 1000.0;

        result[0] = FormatData.getEffectiveValue4Digit(result[0], 3);
        result[1] = FormatData.getEffectiveValue4Digit(result[1], 3);

        return result;
    }
}
