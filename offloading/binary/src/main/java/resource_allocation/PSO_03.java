package resource_allocation;

import config.InitFrame;
import config.RevisePolicy;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import enums.Constants;
import lombok.extern.slf4j.Slf4j;
import utils.FormatData;
import utils.Formula;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@Slf4j(topic = "PSO_03")
public class PSO_03 {

    int numParticles; // 粒子数量
    int numDimensions;   // 粒子维度

    double[] globalBestPos; // 群体最优位置
    double globalBestFitness; // 群体最优适应度
    // double[][] gBestPosK; // 记录 k 个群体最优位置
    // double[] globalBestFitnessK; // k 个群体最优适应度

    Particle[] particles; // 粒子数组

    Particle bestParticle;  // 本轮迭代最优粒子

    // 当前策略
    List<Integer> currUnloadArr;  // 卸载决策
    List<Integer> currFreqAllocArr;  // 资源分配

    public PSO_03() {
    }

    public PSO_03(int numParticles, int numDimensions) {
        this.numParticles = numParticles;
        this.numDimensions = numDimensions;

        List<Integer> initResAllocArr = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            initResAllocArr.add(new Random().nextInt(100) + 1);
        }

        initParticles(numParticles, numDimensions, initResAllocArr);

    }

    public PSO_03(int numParticles, int numDimensions, List<Integer> currInputUnloadArr) {
        this.numParticles = numParticles;
        this.numDimensions = numDimensions;

        this.currUnloadArr = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            currUnloadArr.add(currInputUnloadArr.get(i));
        }

        List<Integer> initResAllocArr = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            initResAllocArr.add(new Random().nextInt(100) + 1);
        }

        initParticles(numParticles, numDimensions, initResAllocArr);

    }

    /**
     * 克隆粒子
     *
     * @param obj
     * @param curr
     * @return
     */
    public Particle cloneParticle(Particle obj, Particle curr) {
        int len = curr.pos.length;
        // double objFitness = curr.
        double[] objPos = new double[len];
        for (int i = 0; i < len; i++) {
            objPos[i] = curr.pos[i];
        }
        obj.pos = objPos;

        obj.pBestFitness = curr.pBestFitness;

        return obj;
    }


    /**
     * 初始化粒子 ==> 根据聚类方案 + 一定随机性
     *
     * @param numParticles
     * @param numDimensions
     * @param resAllocArr
     */
    public void initParticles(int numParticles, int numDimensions, List<Integer> resAllocArr) {
        // numDimensions == tasks.size() == resAlloArr.size()

        this.numParticles = numParticles;
        this.numDimensions = numDimensions;

        particles = new Particle[numParticles];

        bestParticle = new Particle();

        for (int i = 0; i < numParticles; i++) {
            particles[i] = new Particle();
            particles[i].pos = new double[numDimensions];
            particles[i].vel = new double[numDimensions];
            particles[i].pBestPos = new double[numDimensions];

            // 初始化粒子位置和速度
            for (int j = 0; j < numDimensions; j++) {
                // 获取粒子当前的资源分配值(resourceArr[j])
                particles[i].pos[j] = resAllocArr.get(j);
                // TODO:设置速度范围
                particles[i].vel[j] = Math.random() * 2 - 1;
                // 初始化 个体最优位置
                particles[i].pBestPos[j] = particles[i].pos[j];
            }

            // 计算粒子适应度
            double currFitness = evaluateFitness(particles[i]);

            // 更新个体初始适应度
            particles[i].pBestFitness = currFitness;
            // 更新种群最优适应度和位置
            // note: > or <
            if (i == 0 || currFitness < globalBestFitness) {
                globalBestFitness = currFitness;
                globalBestPos = particles[i].pos.clone();
            }
        }

        bestParticle.pBestFitness = globalBestFitness;
        bestParticle.pos = globalBestPos.clone();
    }


    /**
     * TODO: 粒子适应度计算
     *
     * @param particle
     * @return
     */
    public double evaluateFitness(Particle particle) {
        // List<Double> ussList, List<Double> energyList
        /*
        // 应该需要传入 lamdaVar ==>
        double[] currParticlePos = particle.pos;
        // 根据pos，适应度计算公式
        return FormatData.getEffectiveValue4Digit(new Random().nextDouble(), 5);
        // return 0;
         */

        List<Integer> currParticleResAllocArr = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            currParticleResAllocArr.add((int) particle.pos[i]);
        }
        // 当前粒子的卸载决策，可能会被修改
        List<Integer> currParticleUnloadArr = new ArrayList<>();
        for (int i = 0; i < numDimensions; i++) {
            currParticleUnloadArr.add(currUnloadArr.get(i));
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

        // note: 暂定适应度为 ussTotal - theta / energyTotal / (max - min)
        double ussTotal = 0.0;
        double energyTotal = 0.0;
        for (int i = 0; i < numDimensions; i++) {
            ussTotal += currUssList.get(i);
            energyTotal += currEnergyList.get(i);
        }

        double fitness = ussTotal / numDimensions - Constants.USS_THRESHOLD / (energyTotal / numDimensions);

        return FormatData.getEffectiveValue4Digit(1 / fitness, 5);
    }

    /**
     * 迭代优化粒子
     *
     * @param numIterations   ：迭代次数
     * @param inertiaWeight   ： 惯性
     * @param cognitiveWeight ： 学习因子
     * @param socialWeight
     * @param minPos          ： 位置边界
     * @param maxPos
     * @param minVel          ： 速度边界
     * @param maxVel
     * note：根据迭代次数前后期，改变权重的变化
     */
    public void optimizeParticles(int numIterations, double inertiaWeight,
                                  double cognitiveWeight, double socialWeight,
                                  int minPos, int maxPos,
                                  int minVel, int maxVel) {
        // 声明随机数
        Random r = new Random();

        for (int iter = 0; iter < numIterations; iter++) {
            // 更新每个粒子的 pos 和 vel
            for (int i = 0; i < numParticles; i++) {
                for (int j = 0; j < globalBestPos.length; j++) {
                    // update vel
                    particles[i].vel[j] =
                            inertiaWeight * particles[i].vel[j]
                                    + cognitiveWeight * r.nextDouble() * (particles[i].pBestPos[j] - particles[i].pos[j])
                                    + socialWeight * r.nextDouble() * (globalBestPos[j] - particles[i].pos[j]);

                    // TODO: 速度界限限制

                    // update pos
                    particles[i].pos[j] += particles[i].vel[j];

                    // TODO: 位置越界条件判断
                    if (particles[i].pos[j] < minPos) {
                        // ...
                    } else if (particles[i].pos[j] > maxPos) {
                        // ...
                    }
                }

                // 计算粒子更新后的适应度
                double newFitness = evaluateFitness(particles[i]);

                // 更新个体最优适应度和最优位置
                if (newFitness < particles[i].pBestFitness) {
                    // note: > or <
                    particles[i].pBestFitness = newFitness;
                    particles[i].pBestPos = particles[i].pos.clone();
                }

                // 更新种群最优适应度和最优位置
                // note: 多个 globalBestFitness 时，轮盘赌策略 --> 替换
                if (newFitness < globalBestFitness) {
                    globalBestFitness = newFitness;
                    globalBestPos = particles[i].pos.clone();
                }
            }

            // setGlobalBestParticle(obj, globalBestPos);
            // 输出每轮迭代后的最优解
            // log.info("iter:" + (iter + 1)
            //         + ": Best Fitness = " + globalBestFitness
            //         + ": best pos = " + java.util.Arrays.toString(globalBestPos));
        }

        bestParticle.pBestFitness = globalBestFitness;
        bestParticle.pos = globalBestPos.clone();

    }

    // 得到当前最优的个体
    public Particle getBestParticle() {
        return bestParticle;
    }

    // note：根据迭代次数 currIterations 与 总迭代次数，改变权重的变化
    public void optimizeParticles(int numIterations, int currIterations,
                                  double inertiaWeight, double cognitiveWeight, double socialWeight,
                                  int minEdge, int maxEdge) {

    }

    public void optimizeParticles(double inertiaWeight,
                                  double cognitiveWeight, double socialWeight,
                                  int minEdge, int maxEdge) {

    }

}
