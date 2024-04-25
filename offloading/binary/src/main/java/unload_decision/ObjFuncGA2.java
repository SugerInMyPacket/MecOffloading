package unload_decision;

import config.InitFrame;
import config.RevisePolicy;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import sun.management.AgentConfigurationError;
import utils.FormatData;
import utils.Formula;

import java.util.ArrayList;
import java.util.List;

public class ObjFuncGA2 {
    // 根据目标函数进行修改
    static double[] evaluate(List<Integer> currFreqAllocArr, List<Integer> currParticleUnloadArr) {

        int numDimensions = currParticleUnloadArr.size();

        // 双目标函数
        double[] result = new double[2];


        // 当前粒子的卸载决策，可能会被修改
        // List<Integer> currParticleUnloadArr = new ArrayList<>();
        // for (int i = 0; i < numDimensions; i++) {
        //     currParticleUnloadArr.add(unloadArr[i]);
        // }

        // 当前粒子的资源分配方案
        List<Integer> currParticleResAllocArr = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            currParticleResAllocArr.add(currFreqAllocArr.get(i));
        }

        // 资源list
        List<Task> taskList = InitFrame.getTaskList();
        List<Vehicle> vehicleList = InitFrame.getVehicleList();
        RoadsideUnit rsu = InitFrame.getRSU();
        // 计算uss和energy
        List<Double> currUssList = Formula.getUSS4TaskList(taskList, currParticleUnloadArr, currParticleResAllocArr);
        List<Double> currEnergyList = Formula.getEnergy4TaskList(taskList, currParticleUnloadArr, currParticleResAllocArr);

        // 修正卸载
        RevisePolicy.reviseUnloadArr(taskList, vehicleList, rsu, currParticleUnloadArr, currParticleResAllocArr,
                currUssList, currEnergyList);

        // 重新计算uss和energy
        currUssList = Formula.getUSS4TaskList(taskList, currParticleUnloadArr, currParticleResAllocArr);
        currEnergyList = Formula.getEnergy4TaskList(taskList, currParticleUnloadArr, currParticleResAllocArr);

        // USS 总和 & energy 总和
        double ussTotal = 0.0;
        double energyTotal = 0.0;
        for (int i = 0; i < numDimensions; i++) {
            ussTotal += currUssList.get(i);
            energyTotal += currEnergyList.get(i);
        }

        result[0] = ussTotal / (double) numDimensions;
        result[1] = -energyTotal / (double) numDimensions;

        return result;
    }


    static double[] evaluate(int[] unloadArr, List<Integer> currFreqAllocArr) {

        int numDimensions = unloadArr.length;

        // 双目标函数
        double[] result = new double[2];


        // 当前粒子的卸载决策，可能会被修改
        List<Integer> currParticleUnloadArr = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            currParticleUnloadArr.add(unloadArr[i]);
        }

        // 当前粒子的资源分配方案
        List<Integer> currParticleResAllocArr = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            currParticleResAllocArr.add(currFreqAllocArr.get(i));
        }

        // 资源list
        List<Task> taskList = InitFrame.getTaskList();
        List<Vehicle> vehicleList = InitFrame.getVehicleList();
        RoadsideUnit rsu = InitFrame.getRSU();

        // 计算uss和energy
        List<Double> currUssList = new ArrayList<>();
        List<Double> currEnergyList = new ArrayList<>();
        // currUssList = Formula.getUSS4TaskList(taskList, currParticleUnloadArr, currParticleResAllocArr);
        // currEnergyList = Formula.getEnergy4TaskList(taskList, currParticleUnloadArr, currParticleResAllocArr);

        // 修正卸载
        // RevisePolicy.reviseUnloadArr(taskList, vehicleList, rsu, currParticleUnloadArr, currParticleResAllocArr,
        //         currUssList, currEnergyList);

        RevisePolicy.reviseUnloadArr(taskList, vehicleList, rsu, currParticleUnloadArr, currParticleResAllocArr);

        // NOTE: 将修正后的 unloadArr 赋值
        for (int j = 0; j < numDimensions; j++) {
            unloadArr[j] = currParticleUnloadArr.get(j);
        }

        // 重新计算uss和energy
        currUssList = Formula.getUSS4TaskList(taskList, currParticleUnloadArr, currParticleResAllocArr);
        currEnergyList = Formula.getEnergy4TaskList(taskList, currParticleUnloadArr, currParticleResAllocArr);

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
