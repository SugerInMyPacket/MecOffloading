package entity;

import lombok.extern.slf4j.Slf4j;

import java.util.List;

@Slf4j
public class Vehicle {

    private int vehicleID;

    // Vehicle 剩余处理能力
    private long freqRemain;
    // Vehicle 最大处理能力
    private long freqMax;

    private double posX;
    private double posY;

    // 车辆的任务集
    private List<Task> taskList;

    public Vehicle() {}

    public Vehicle(long freqRemain, long freqMax) {
        this.freqRemain = freqRemain;
        this.freqMax = freqMax;
    }

    public Vehicle(int vehicleID, long freqRemain, long freqMax) {
        this.vehicleID = vehicleID;
        this.freqRemain = freqRemain;
        this.freqMax = freqMax;
    }

    public int getVehicleID() {
        return vehicleID;
    }

    public void setVehicleID(int vehicleID) {
        this.vehicleID = vehicleID;
    }

    public long getFreqRemain() {
        return freqRemain;
    }

    public void setFreqRemain(long freqRemain) {
        if (freqRemain > freqMax) {
            this.freqRemain = freqMax;
        } else {
            this.freqRemain = freqRemain;
        }
    }

    public long getFreqMax() {
        return freqMax;
    }

    public void setFreqMax(long freqMax) {
        this.freqMax = freqMax;
    }

    public double getPosX() {
        return posX;
    }

    public void setPosX(double posX) {
        this.posX = posX;
    }

    public double getPosY() {
        return posY;
    }

    public void setPosY(double posY) {
        this.posY = posY;
    }

    public List<Task> getTaskList() {
        return taskList;
    }

    public void setTaskList(List<Task> taskList) {
        this.taskList = taskList;
    }

    @Override
    public String toString() {
        return "Vehicle{" +
                "vehicleID=" + vehicleID +
                ", freqRemain=" + freqRemain +
                ", freqMax=" + freqMax +
                ", posX=" + posX +
                ", posY=" + posY +
                '}';
    }


}
