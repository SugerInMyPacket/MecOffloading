package classify;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class KMeansClustering {

    public static void main(String[] args) {
        try {
            // 读取数据
            // DataSource source = new DataSource("path/to/your/dataset.arff");
            DataSource source = new DataSource("data/xxx.arff");
            Instances data = source.getDataSet();

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
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
