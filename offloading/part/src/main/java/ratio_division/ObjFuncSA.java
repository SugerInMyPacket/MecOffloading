package ratio_division;

import config.InitFrame;
import config.RevisePolicy;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import utils.FormatData;
import utils.Formula;

import java.util.ArrayList;
import java.util.List;

public class ObjFuncSA {

    static double[] evaluate(List<Integer> currParticleUnloadArr,
                             double[] unloadRatioArr,
                             List<Integer> currParticleFreqAllocArr) {

        double[] result = new double[2];


        return result;
    }

    static double[] evaluate(List<Integer> currParticleUnloadArr,
                             double[] unloadRatioArr,
                             List<Integer> currParticleFreqAllocArrLocal,
                             List<Integer> currParticleFreqAllocArrRemote) {

        double[] result = new double[2];

        int numDimensions = unloadRatioArr.length;

        List<Double> currParticleUnloadRatioArr = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            currParticleUnloadRatioArr.add(unloadRatioArr[i]);
        }

        // 资源list
        // List<Task> taskList = InitFrame.getTaskList();
        List<Vehicle> vehicleList = InitFrame.getVehicleList();
        RoadsideUnit rsu = InitFrame.getRSU();

        List<Double> currUssList = new ArrayList<>();
        List<Double> currEnergyList = new ArrayList<>();

        // 计算uss和energy
        // currUssList = Formula.getUSS4TaskList(taskList, currParticleUnloadArr,
        //         unloadRatioArr, currFreqAllocArrLocal, currFreqAllocArrRemote);
        // currEnergyList = Formula.getEnergy4TaskList(taskList, currParticleUnloadArr,
        //         unloadRatioArr, currFreqAllocArrLocal, currFreqAllocArrRemote);

        // 修正卸载
        // RevisePolicy.reviseUnloadArrRemote(taskList, vehicleList, rsu,
        //         currParticleUnloadArr, currFreqAllocArrRemote,
        //         currUssList, currEnergyList);

        // RevisePolicy.reviseUnloadArrRemote(taskList, vehicleList, rsu,
        //         currParticleUnloadArr, currFreqAllocArrRemote);

        // RevisePolicy.reviseUnloadArrRemote(taskList, vehicleList, rsu,
        //         currParticleUnloadArr, currFreqAllocArrRemote);


        List<Task> taskList = InitFrame.getTaskList();
        // 将修正后的 unloadArr 赋值
        // for (int j = 0; j < numDimensions; j++) {
        //     unloadArr[j] = currParticleUnloadArr.get(j);
        // }
        for (int i = 0; i < numDimensions; i++) {
            unloadRatioArr[i] = currParticleUnloadRatioArr.get(i);
        }

        // 重新计算uss和energy
        currUssList = Formula.getUSS4TaskList(taskList,
                currParticleUnloadArr,
                currParticleUnloadRatioArr,
                currParticleFreqAllocArrLocal,
                currParticleFreqAllocArrRemote);
        currEnergyList
                = Formula.getEnergy4TaskList(taskList,
                currParticleUnloadArr,
                currParticleUnloadRatioArr,
                currParticleFreqAllocArrLocal,
                currParticleFreqAllocArrRemote);

        // USS 总和 & energy 总和
        double ussTotal = 0.0;
        double energyTotal = 0.0;
        for (int i = 0; i < numDimensions; i++) {
            ussTotal += currUssList.get(i);
            energyTotal += currEnergyList.get(i);
        }

        result[0] = ussTotal / (double) numDimensions;
        result[1] = -energyTotal / (double) numDimensions / 1000.0;

        // 规范化
        result[0] = FormatData.getEffectiveValue4Digit(result[0], 3);
        result[1] = FormatData.getEffectiveValue4Digit(result[1], 3);

        return result;
    }
}
