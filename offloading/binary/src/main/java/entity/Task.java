package entity;

import lombok.Data;

@Data
public class Task {

    // 在进行schedule时任务的id（编号）
    private int taskID;
    // 任务所属车辆的id（编号）
    private int vehicleID;

    // size : 任务大小 (bit)
    private int s;
    // rate : 输入数据与输出数据大小的比值
    private float r;
    // 处理密度（单位：CPU cycles/ bit ）
    private float c;
    // deadline：截止期
    private double d;
    // 截止期限度系数（容忍度因子）
    private int factor;
    // 类别
    private int I;
    // 优先级
    private int p;

    // 任务聚类类别
    private int clusterID;

    public int getClusterID() {
        return clusterID;
    }

    public void setClusterID(int clusterID) {
        this.clusterID = clusterID;
    }

    @Override
    public String toString() {
        return "Task{" +
                "taskID=" + taskID +
                ", size=" + s +
                ", rate=" + r +
                ", c=" + c +
                ", deadline=" + d +
                ", factor=" + factor +
                ", I=" + I +
                ", prior=" + p +
                ", vehicleID=" + vehicleID +
                '}';
    }

    public int getS() {
        return s;
    }

    public void setS(int s) {
        this.s = s;
    }

    public float getR() {
        return r;
    }

    public void setR(float r) {
        this.r = r;
    }

    public float getC() {
        return c;
    }

    public void setC(float c) {
        this.c = c;
    }

    public double getD() {
        return d;
    }

    public void setD(long d) {
        this.d = d;
    }

    public int getFactor() {
        return factor;
    }

    public void setFactor(int factor) {
        this.factor = factor;
    }

    public int getI() {
        return I;
    }

    public void setI(int i) {
        I = i;
    }

    public int getP() {
        return p;
    }

    public void setP(int p) {
        this.p = p;
    }
}
