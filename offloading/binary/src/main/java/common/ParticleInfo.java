package common;

// 粒子

import entity.Cloud;
import entity.RoadsideUnit;
import entity.Vehicle;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;

@Slf4j
public class ParticleInfo {
    // 卸载决策 : 任务i 卸载到 目标j
    List<Integer> lamdaVar;

    // f[i][j] ==> 编号为 i 的任务 卸载到 车辆 j 后 得到的计算资源为 q
    // List<List<Integer>> freqOfv4t;
    List<Integer> freqOfv4t;

    List<Vehicle> vehicleList;
    RoadsideUnit rsu;
    Cloud c;

    @Override
    public String toString() {
        int lamdaVarSize = lamdaVar.size();
        // int freqOfv4tSize = freqOfv4t.size();
        int vehicleSize = vehicleList.size();

        int[] lamdaVarArr = new int[lamdaVarSize];
        for (int i = 0; i < lamdaVarSize; i++) {
            lamdaVarArr[i] = lamdaVar.get(i);
        }

        Long[] vehicleCapArr = new Long[vehicleSize];
        for (int i = 0; i < vehicleSize; i++) {
            vehicleCapArr[i] = vehicleList.get(i).getFreqRemain();
        }

        return "Particle{" +
                "lamdaVar=" + lamdaVarArr +
                ", freqOfv4t=" + freqOfv4t +
                ", vehicleCapList=" + vehicleCapArr +
                ", rsu=" + rsu +
                ", c=" + c +
                '}';
    }

    public ParticleInfo() {
        int lamdaSize = 0;
        for(Vehicle v : vehicleList) {
            lamdaSize += v.getTaskList().size();
        }
        lamdaVar = new ArrayList<>(lamdaSize);
    }


    public List<Integer> getLamdaVar() {
        return lamdaVar;
    }

    public void setLamdaVar(List<Integer> lamdaVar) {
        this.lamdaVar = lamdaVar;
    }

    public List<Integer> getFreqOfv4t() {
        return freqOfv4t;
    }

    public void setFreqOfv4t(List<Integer> freqOfv4t) {
        this.freqOfv4t = freqOfv4t;
    }

    public List<Vehicle> getVehicleList() {
        return vehicleList;
    }

    public void setVehicleList(List<Vehicle> vehicleList) {
        this.vehicleList = vehicleList;
    }

    public RoadsideUnit getRsu() {
        return rsu;
    }

    public void setRsu(RoadsideUnit rsu) {
        this.rsu = rsu;
    }

    public Cloud getC() {
        return c;
    }

    public void setC(Cloud c) {
        this.c = c;
    }
}
