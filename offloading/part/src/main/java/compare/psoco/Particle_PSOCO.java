package compare.psoco;

import utils.ArrayUtils;
import utils.NumUtil;

import java.util.Random;

public class Particle_PSOCO {
    // 坐标
    double[] posUnload;
    double[] posUnloadRatio;
    double[] posFreqLocal;
    double[] posFreqRemote;

    // 速度
    double[] velUnload;
    double[] velUnloadRatio;
    double[] velFreqLocal;
    double[] velFreqRemote;
    // 个体最佳位置
    double[] pBestPosition;
    // 最佳适应度
    double[] pBestFitness;

    public Particle_PSOCO() {}

    /***
     * 初始化粒子
     * @param dim
     * @param bound1 : 卸载决策边界
     * @param bound2 : 卸载比率边界
     * @param bounds3 : 本地freq边界
     * @param bounds4 : 远程freq边界
     */
    public Particle_PSOCO(int dim, int bound1, double[][] bounds3, double[][] bounds4) {
        Random random = new Random();

        posUnload = new double[dim];
        posUnloadRatio = new double[dim];
        posFreqLocal = new double[dim];
        posFreqRemote = new double[dim];

        velUnload = new double[dim];
        velUnloadRatio = new double[dim];
        velFreqLocal = new double[dim];
        velFreqRemote = new double[dim];

        pBestPosition = new double[dim * 4];
        pBestFitness = new double[2];

        for (int i = 0; i < dim; i++) {
            posUnload[i] = -1 + random.nextDouble() * (bound1 + 1);
            velUnload[i] = -1 + random.nextDouble() * (bound1 + 1);

            posUnloadRatio[i] = NumUtil.random(0.0, 1.0);
            velUnloadRatio[i] = NumUtil.random(0.0, 1.0);

            posFreqLocal[i] = bounds3[i][0] + random.nextDouble() * (bounds3[i][1] - bounds3[i][0]);
            velFreqLocal[i] = bounds3[i][0] + random.nextDouble() * (bounds3[i][1] - bounds3[i][0]);

            posFreqRemote[i] = bounds4[i][0] + random.nextDouble() * (bounds4[i][1] - bounds4[i][0]);
            velFreqRemote[i] = bounds4[i][0] + random.nextDouble() * (bounds4[i][1] - bounds4[i][0]);

            pBestPosition = ArrayUtils.connectArrays(posUnload, posUnloadRatio, posFreqLocal, posFreqRemote);
        }

        pBestFitness = ObjFunc_PSOCO.evaluate(posUnload, posUnloadRatio, posFreqLocal, posFreqRemote);

    }

}
