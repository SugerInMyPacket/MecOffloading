package entity;

import java.util.List;

public class RoadsideUnit {
    private long freqRemain;
    private long freqMax;

    // RSU的车辆集
    private List<Vehicle> vehicleList;

    public long getFreqRemain() {
        return freqRemain;
    }

    public void setFreqRemain(long freqRemain) {
        this.freqRemain = freqRemain;
    }

    public long getFreqMax() {
        return freqMax;
    }

    public void setFreqMax(long freqMax) {
        this.freqMax = freqMax;
    }

    public List<Vehicle> getVehicleList() {
        return vehicleList;
    }

    public void setVehicleList(List<Vehicle> vehicleList) {
        this.vehicleList = vehicleList;
    }
}
