package utils;

import entity.Task;
import entity.Vehicle;

import java.io.BufferedReader;
import java.io.FileReader;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

public class FileUtils {

    public static List<Task> taskList = new ArrayList<>();

    public static List<Vehicle> vehicleList = new ArrayList<>();

    public static void main(String[] args) throws Exception {
        // task 数据写入
        // readFileByLine("../data/task01.csv");
        // showTaskList();
        // insertTaskData2DB();

        // vehicle 数据写入
        readFileByLine4Vehicle("../data/vehicle01.csv");
        showVehicleList();
        insertVehicleData2DB();
    }

    /**
     * 读取文件
     *
     * @param filepath
     * @throws Exception
     */
    public static void readFileByLine(String filepath) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader(filepath));
        String line = null;
        // 跳过头部标题行
        if ((line = reader.readLine()) == null) {
            reader.close();
            return;
        }
        // 遍历读取数据
        while ((line = reader.readLine()) != null) {
            // System.out.println(line); //第一行

            Task task = new Task();
            String[] taskStr = line.split(",");
            task.setTaskID(Integer.parseInt(taskStr[0]));
            task.setS(Integer.parseInt(taskStr[1]));
            task.setR(Float.parseFloat(taskStr[2]));
            task.setC(Float.parseFloat(taskStr[3]));
            task.setD(Integer.parseInt(taskStr[4]));
            task.setFactor(Integer.parseInt(taskStr[5]));
            task.setI(Integer.parseInt(taskStr[6]));
            task.setP(Integer.parseInt(taskStr[7]));
            task.setVehicleID(Integer.parseInt(taskStr[8]));

            taskList.add(task);
        }
        reader.close();
    }


    public static void readFileByLine4Vehicle(String filepath) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader(filepath));
        String line = null;
        // 跳过头部标题行
        if ((line = reader.readLine()) == null) {
            reader.close();
            return;
        }
        // 遍历读取数据
        while ((line = reader.readLine()) != null) {
            // System.out.println(line); //第一行

            Vehicle vehicle = new Vehicle();
            String[] vehicleStr = line.split(",");
            vehicle.setVehicleID(Integer.parseInt(vehicleStr[0]));
            vehicle.setFreqMax(Integer.parseInt(vehicleStr[2]));
            vehicle.setFreqRemain(Integer.parseInt(vehicleStr[1]));
            vehicle.setPosX(Double.parseDouble(vehicleStr[3]));
            vehicle.setPosY(Double.parseDouble(vehicleStr[4]));

            vehicleList.add(vehicle);
        }
        reader.close();
    }


    public static void insertTaskData2DB() throws SQLException {

        Connection conn = null;
        // 驱动程序名
        String driver = "com.mysql.cj.jdbc.Driver";
        // URL指向要访问的数据库名mec
        String url = "jdbc:mysql://localhost:3306/mec?serverTimezone=GMT%2B8&useUnicode=true&characterEncoding=utf8&autoReconnect=true&allowMultiQueries=true&useSSL=false";
        // MySQL配置时的用户名
        String user = "root";
        // MySQL配置时的密码
        String password = "123456";
        // 遍历查询结果集
        // 加载驱动程序
        try {
            Class.forName(driver);
            // 连接MySQL数据库！！
            conn = DriverManager.getConnection(url, user, password);
            conn.setAutoCommit(false);
            // note: 注意数据插入
            String insertSQLstr = "INSERT INTO task(task_id, size, rate, c, deadline, factor, kind, prior, vehicle_id) VALUES (?,?,?,?,?,?,?,?,?)";
            PreparedStatement prep = conn.prepareStatement(insertSQLstr); //需要替换
            int num = 0;
            for (Task task : taskList) {
                num++;
                prep.setString(1, String.valueOf(task.getTaskID()));
                prep.setString(2, String.valueOf(task.getS()));
                prep.setString(3, String.valueOf(task.getR()));
                prep.setString(4, String.valueOf(task.getC()));
                prep.setString(5, String.valueOf(task.getD()));
                prep.setString(6, String.valueOf(task.getFactor()));
                prep.setString(7, String.valueOf(task.getI()));
                prep.setString(8, String.valueOf(task.getP()));
                prep.setString(9, String.valueOf(task.getVehicleID()));
                prep.addBatch();
                if (num > 50000) {
                    System.out.println(prep);
                    prep.executeBatch();
                    conn.commit();
                    num = 0;
                }
                System.out.println(prep);
                prep.executeBatch();
                conn.commit();
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                conn.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }

    }

    public static void insertVehicleData2DB() throws SQLException {

        Connection conn = null;
        // 驱动程序名
        String driver = "com.mysql.cj.jdbc.Driver";
        // URL指向要访问的数据库名mec
        String url = "jdbc:mysql://localhost:3306/mec?serverTimezone=GMT%2B8&useUnicode=true&characterEncoding=utf8&autoReconnect=true&allowMultiQueries=true&useSSL=false";
        // MySQL配置时的用户名
        String user = "root";
        // MySQL配置时的密码
        String password = "123456";
        // 遍历查询结果集
        // 加载驱动程序
        try {
            Class.forName(driver);
            // 连接MySQL数据库！！
            conn = DriverManager.getConnection(url, user, password);
            conn.setAutoCommit(false);
            // note: 注意数据插入
            String insertSQLstr = "INSERT INTO vehicle(vehicle_id, freq_remain, freq_max, pos_x, pos_y) VALUES (?,?,?,?,?)";
            PreparedStatement prep = conn.prepareStatement(insertSQLstr); //需要替换
            int num = 0;
            for (Vehicle vehicle : vehicleList) {
                num++;
                prep.setString(1, String.valueOf(vehicle.getVehicleID()));
                prep.setString(2, String.valueOf(vehicle.getFreqRemain()));
                prep.setString(3, String.valueOf(vehicle.getFreqMax()));
                prep.setString(4, String.valueOf(vehicle.getPosX()));
                prep.setString(5, String.valueOf(vehicle.getPosY()));
                prep.addBatch();
                if (num > 50000) {
                    System.out.println(prep);
                    prep.executeBatch();
                    conn.commit();
                    num = 0;
                }
                System.out.println(prep);
                prep.executeBatch();
                conn.commit();
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                conn.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }

    }

    public static void showTaskList() {
        int len = taskList.size();
        for (int i = 0; i < len; i++) {
            System.out.println(i + ": " + taskList.get(i));
        }
    }

    public static void showVehicleList() {
        int len = vehicleList.size();
        for (int i = 0; i < len; i++) {
            System.out.println(i + ": " + vehicleList.get(i));
        }
    }

}
