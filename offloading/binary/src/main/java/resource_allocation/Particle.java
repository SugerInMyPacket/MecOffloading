package resource_allocation;

import java.util.List;

public class Particle {
    double[] pos; // 粒子当前位置
    double[] vel; // 粒子当前速度

    double[] pBestPos; // 粒子个体最优位置
    double pBestFitness; // 粒子个体最优适应度

    // keynote: 重构 --> 参数的修改使用 setter()


    // 计算粒子适应度
    public static double getFitness(List<Integer> unloadArr, List<Integer> resArr) {

        return 0;
    }

    public double[] getPos() {
        return pos;
    }

    public void setPos(double[] pos) {
        this.pos = pos;
    }

    public double[] getVel() {
        return vel;
    }

    public void setVel(double[] vel) {
        this.vel = vel;
    }

    public double[] getpBestPos() {
        return pBestPos;
    }

    public void setpBestPos(double[] pBestPos) {
        this.pBestPos = pBestPos;
    }

    public double getpBestFitness() {
        return pBestFitness;
    }

    public void setpBestFitness(double pBestFitness) {
        this.pBestFitness = pBestFitness;
    }
}
