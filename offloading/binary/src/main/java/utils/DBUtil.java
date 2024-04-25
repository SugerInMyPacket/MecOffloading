package utils;

import java.io.IOException;
import java.sql.*;
import java.util.Properties;

public class DBUtil {
    static Properties properties = null; // 用于读取和处理资源文件中的信息

    static { // 类加载的时候被执行一次
        properties = new Properties();
        try {
            properties.load(Thread.currentThread().getContextClassLoader().getResourceAsStream("db.properties"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static Connection getConnection() {
        try {
            // 加载 MySQL JDBC 驱动类
            Class.forName(properties.getProperty("mysqlDriver"));
            // 建立连接（连接对象内部其实包含了Socket对象，是一个远程的连接，比较耗时！这是Connection对象管理的一个要点！）
            // 真正开发中，为了提高效率，都会使用连接池来管理连接对象！
            String mysqlUrl = properties.getProperty("mysqlUrl");
            String mysqlUser = properties.getProperty("mysqlUser");
            String mysqlPassword = properties.getProperty("mysqlPassword");
            return DriverManager.getConnection(mysqlUrl, mysqlUser, mysqlPassword);
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        } catch (SQLException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static PreparedStatement getPreparedStatement(Connection connection, String sql) {
        try {
            // 使用 PreparedStatement，防止 SQL 注入
            return connection.prepareStatement(sql);
        } catch (SQLException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static void close(Connection connection, Statement statement, ResultSet resultSet) {
        if (resultSet != null) {
            try {
                resultSet.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
        if (statement != null) {
            try {
                statement.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
        if (connection != null) {
            try {
                connection.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }

    public static void close(Connection connection) {
        if (connection != null) {
            try {
                connection.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }

    public static void close(Statement statement) {
        if (statement != null) {
            try {
                statement.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }

    public static void close(ResultSet resultSet) {
        if (resultSet != null) {
            try {
                resultSet.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }

}
