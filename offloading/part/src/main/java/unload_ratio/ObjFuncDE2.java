package unload_ratio;

import config.InitFrame;
import config.RevisePolicy;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import utils.FormatData;
import utils.Formula;

import java.util.ArrayList;
import java.util.List;

public class ObjFuncDE2 {
    // 目标函数计算
    static double[] evaluate(int[] unloadArr,
                             double[] ratioArr,
                             List<Integer> currFreqAllocArrLocal,
                             List<Integer> currFreqAllocArrRemote) {

        int numDimensions = unloadArr.length;

        // 双目标函数
        double[] result = new double[2];


        // 当前粒子的卸载决策，可能会被修改
        List<Integer> currParticleUnloadArr = new ArrayList<>();
        List<Double> currParticleRatioArr = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            currParticleUnloadArr.add(unloadArr[i]);
            currParticleRatioArr.add(ratioArr[i]);
        }


        // 当前粒子的资源分配方案
        // List<Integer> currParticleResAllocArr = new ArrayList<>();
        // for (int i = 0; i < numDimensions; i++) {
        //     currParticleResAllocArr.add(currFreqAllocArr.get(i));
        // }

        // 资源list
        List<Task> taskList = InitFrame.getTaskList();
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

        // 此处仅仅修正了Remote
        RevisePolicy.reviseUnloadArrRemote(taskList, vehicleList, rsu,
                currParticleUnloadArr, currFreqAllocArrRemote);

        // KEYNOTE：TODO：Remote修正unloadArr， Local修正freqLocal

        // 将修正后的 unloadArr 赋值
        for (int j = 0; j < numDimensions; j++) {
            unloadArr[j] = currParticleUnloadArr.get(j);
        }

        // 重新计算uss和energy
        currUssList = Formula.getUSS4TaskList(taskList, currParticleUnloadArr, currParticleRatioArr,
                currFreqAllocArrLocal, currFreqAllocArrRemote);
        currEnergyList = Formula.getEnergy4TaskList(taskList, currParticleUnloadArr, currParticleRatioArr,
                currFreqAllocArrLocal, currFreqAllocArrRemote);

        // USS 总和 & energy 总和
        double ussTotal = 0.0;
        double energyTotal = 0.0;
        for (int i = 0; i < numDimensions; i++) {
            ussTotal += currUssList.get(i);
            energyTotal += currEnergyList.get(i);
        }

        result[0] = ussTotal / (double) numDimensions;
        result[1] = -energyTotal / (double) numDimensions / 100000.0;

        // 规范化
        result[0] = FormatData.getEffectiveValue4Digit(result[0], 3);
        result[1] = FormatData.getEffectiveValue4Digit(result[1], 3);

        return result;
    }

    static double[] evaluate(double[] solutionAll,
                             List<Integer> currFreqAllocArrLocal,
                             List<Integer> currFreqAllocArrRemote) {

        int numDimensions = solutionAll.length / 2;

        // 双目标函数
        double[] result = new double[2];


        // 当前粒子的卸载决策，可能会被修改
        List<Integer> currParticleUnloadArr = new ArrayList<>();
        List<Double> currParticleRatioArr = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            currParticleUnloadArr.add((int) solutionAll[i]);
            currParticleRatioArr.add(solutionAll[i + numDimensions]);
        }

        // 资源list
        List<Task> taskList = InitFrame.getTaskList();
        List<Vehicle> vehicleList = InitFrame.getVehicleList();
        RoadsideUnit rsu = InitFrame.getRSU();

        List<Double> currUssList = new ArrayList<>();
        List<Double> currEnergyList = new ArrayList<>();

        // 此处仅仅修正了Remote
        RevisePolicy.reviseUnloadArrRemote(taskList, vehicleList, rsu,
                currParticleUnloadArr, currFreqAllocArrRemote);

        // KEYNOTE：TODO：Remote修正unloadArr， Local修正freqLocal

        // 将修正后的 unloadArr 赋值
        for (int j = 0; j < numDimensions; j++) {
            solutionAll[j] = currParticleUnloadArr.get(j);
        }

        // 重新计算uss和energy
        currUssList = Formula.getUSS4TaskList(taskList, currParticleUnloadArr, currParticleRatioArr,
                currFreqAllocArrLocal, currFreqAllocArrRemote);
        currEnergyList = Formula.getEnergy4TaskList(taskList, currParticleUnloadArr, currParticleRatioArr,
                currFreqAllocArrLocal, currFreqAllocArrRemote);

        // USS 总和 & energy 总和
        double ussTotal = 0.0;
        double energyTotal = 0.0;
        for (int i = 0; i < numDimensions; i++) {
            ussTotal += currUssList.get(i);
            energyTotal += currEnergyList.get(i);
        }

        result[0] = ussTotal / (double) numDimensions;
        result[1] = -energyTotal / (double) numDimensions / 100000.0;

        // 规范化
        result[0] = FormatData.getEffectiveValue4Digit(result[0], 3);
        result[1] = FormatData.getEffectiveValue4Digit(result[1], 3);

        return result;
    }

}
