package classify;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;

public class KMeansClusteringFromDB {

    public static void main(String[] args) {
        try {
            // 连接数据库
            String jdbcUrl = "jdbc:mysql://localhost:3306/mec";
            String username = "root";
            String password = "123456";
            Connection connection = DriverManager.getConnection(jdbcUrl, username, password);

            // 查询数据
            Statement statement = connection.createStatement();
            // note：将需要用到的 attribute 提出
            ResultSet resultSet = statement.executeQuery("SELECT attribute1, attribute2, attribute3, ..., attributeN FROM your_table");

            // 从数据库结果集创建ARFF文件
            // 这只是示例，具体的实现可能需要根据数据库结构进行调整
            Instances data = DataSource.read(String.valueOf(resultSet));

            // 设置簇的数量为4
            SimpleKMeans kMeans = new SimpleKMeans();
            kMeans.setNumClusters(4);

            // 进行聚类
            kMeans.buildClusterer(data);

            // 获取聚类结果
            int[] assignments = kMeans.getAssignments();

            // 打印每个实例的类别
            for (int i = 0; i < assignments.length; i++) {
                System.out.println("任务" + (i + 1) + "的类别: " + assignments[i]);
            }

            // 关闭数据库连接
            resultSet.close();
            statement.close();
            connection.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
