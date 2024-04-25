package config;

import entity.Cloud;
import entity.RoadsideUnit;
import entity.Task;
import entity.Vehicle;
import enums.Constants;
import utils.DBUtil;
import utils.Formula;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class InitFrame {

    public static void main(String[] args) {
        new InitFrame();
        int tasklen = taskList.size();
        for (int i = 0; i < tasklen; i++) {
            System.out.println(taskList.get(i));
        }

        int vehicleLen = vehicleList.size();
        for (int i = 0; i < vehicleLen; i++) {
            System.out.println(vehicleList.get(i));
        }

    }

    static int len = 10;
    static int vehicleNums = Constants.VEHICLE_NUMS;

    public static List<Task> taskList;

    public static List<Vehicle> vehicleList;

    public static RoadsideUnit rsu;

    public static Cloud cloud;

    public InitFrame() {
        // init();
        initFromDB();
    }

    public static void init() {
        taskList = new ArrayList<>();
        vehicleList = new ArrayList<>();
        rsu = new RoadsideUnit();
        cloud = new Cloud();

        initTaskList();
        initAllRes();
    }

    public static void initFromDB() {
        taskList = new ArrayList<>();
        vehicleList = new ArrayList<>();
        rsu = new RoadsideUnit();
        cloud = new Cloud();

        initTaskListFromDB();
        initAllResFromDB();
    }

    public static void initAllResFromDB() {

        // Cloud
        cloud.setFreqMax(1000000);
        cloud.setFreqRemain(Constants.FREQ_REMAIN_Cloud);


        // 初始化 RSU 的信息
        rsu.setFreqMax(50000);
        rsu.setFreqRemain(Constants.FREQ_REMAIN_RSU);

        initVehicleListFromDB();


        // ------------------ 初始化车辆信道增益
        List<Double> gainChannelVehicleList = new ArrayList<>();
        for (int i = 0; i < vehicleNums; i++) {
            // gainChannelVehicleList.add(new Random().nextDouble());
            gainChannelVehicleList.add(1.0);
        }
        Formula.initGainChannelVehicles(gainChannelVehicleList);

    }

    public static void initAllRes() {
        // Cloud
        cloud.setFreqMax(50000);
        cloud.setFreqRemain(30000);

        // 初始化 RSU 的信息
        rsu.setFreqMax(10000);
        rsu.setFreqRemain(500);

        // 初始化车辆
        for (int i = 0; i < vehicleNums; i++) {
            Vehicle vehicle = new Vehicle();
            vehicle.setVehicleID(i + 1);
            vehicle.setFreqMax(200);
            vehicle.setFreqRemain(200);

            vehicleList.add(vehicle);
        }

        // ------------------ 初始化车辆信道增益
        List<Double> gainChannelVehicleList = new ArrayList<>();
        for (int i = 0; i < vehicleNums; i++) {
            // gainChannelVehicleList.add(new Random().nextDouble());
            gainChannelVehicleList.add(1.0);
        }
        Formula.initGainChannelVehicles(gainChannelVehicleList);

    }

    public static void initVehicleListFromDB() {
        Connection conn = DBUtil.getConnection();
        String sql = "select * from  " + Constants.VehicleDB;

        PreparedStatement ps = DBUtil.getPreparedStatement(conn, sql);
        ResultSet rs = null;
        try {
            // 返回查询结果
            rs = ps.executeQuery();

            while (rs.next()) {
                Vehicle vehicle = new Vehicle();
                vehicle.setVehicleID(rs.getInt("vehicle_id"));
                vehicle.setFreqMax(rs.getInt("freq_max"));
                vehicle.setFreqRemain(rs.getInt("freq_remain"));
                vehicle.setPosX(rs.getInt("pos_x"));
                vehicle.setPosY(rs.getInt("pos_y"));
                vehicleList.add(vehicle);
            }

        } catch (SQLException e) {
            e.printStackTrace();
        }
    }


    // 初始化任务list
    public static void initTaskList() {
        for (int i = 0; i < len; i++) {
            Task newTask = new Task();
            newTask.setTaskID(1000 + i);
            newTask.setS(100);
            newTask.setR(0.1f);
            newTask.setC(10);
            newTask.setD(1000);
            newTask.setFactor(5);
            newTask.setI(1);
            newTask.setP(3);
            newTask.setVehicleID(new Random().nextInt(vehicleNums));
            taskList.add(newTask);
        }
    }

    public static void initTaskListFromDB() {
        Connection conn = DBUtil.getConnection();
        String sql = "select * from  " + Constants.TaskDB;

        PreparedStatement ps = DBUtil.getPreparedStatement(conn, sql);
        ResultSet rs = null;
        try {
            // 返回查询结果
            rs = ps.executeQuery();

            while (rs.next()) {
                Task task = new Task();
                task.setTaskID(rs.getInt("task_id"));
                task.setS((int) (rs.getInt("size") * Constants.DATA_SIZE_MULTI_INCREASE));
                task.setR(rs.getFloat("rate"));
                task.setC(rs.getFloat("c"));
                task.setD(rs.getInt("deadline"));
                task.setFactor(rs.getInt("factor"));
                task.setI(rs.getInt("kind"));
                task.setP(rs.getInt("prior"));
                task.setVehicleID(rs.getInt("vehicle_id"));
                task.setClusterID(rs.getInt("cluster_class"));
                // task.setTaskID(rs.getInt("task_id"));
                taskList.add(task);
            }

        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public static List<Task> getTaskList() {
        return taskList;
    }


    // 初始化车辆list，资源信息等
    public static void initVehicleList() {

    }

    public static List<Vehicle> getVehicleList() {
        return vehicleList;
    }

    public static void initRSU() {

    }

    public static RoadsideUnit getRSU() {
        return rsu;
    }

    public static void initCloud() {

    }

    public static Cloud getCloud() {
        return cloud;
    }
}
