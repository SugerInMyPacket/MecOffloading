package algorithm;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class GeneticParticleAlgorithm {
    static Random random = new Random();

    // 遗传算法更新前k位
    public static List<Integer> updateGenetic(List<Integer> list, int k, int m) {
        // 实现遗传算法更新前k位的逻辑
        // ...
        return list.subList(0, k); // 返回更新后的前k位列表
    }

    // 粒子群算法更新后k位
    public static List<Integer> updateParticle(List<Integer> list, int k, int n) {
        // 实现粒子群算法更新后k位的逻辑
        // ...
        return list.subList(k, 2 * k); // 返回更新后的后k位列表
    }

    // 随机生成长度为2k的列表
    public static List<Integer> generateList(int k, int m, int n) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            list.add(random.nextInt(m)); // 前k位的值范围是0到m
        }
        for (int i = 0; i < k; i++) {
            list.add(random.nextInt(n)); // 后k位的值范围是0到n
        }
        return list;
    }

    public static void main(String[] args) {
        int k = 5; // 设置k的值
        int m = 10; // 设置m的值
        int n = 20; // 设置n的值

        List<List<Integer>> lists = new ArrayList<>(); // 用于存储多个列表

        // 生成多个长度为2k的列表
        for (int i = 0; i < 5; i++) { // 这里生成5个列表，你可以根据需要修改数量
            List<Integer> list = generateList(k, m, n);
            lists.add(list);
        }

        // 遍历列表，分别使用遗传算法和粒子群算法进行更新
        for (List<Integer> list : lists) {
            List<Integer> updatedGenetic = updateGenetic(list, k, m);
            List<Integer> updatedParticle = updateParticle(list, k, n);

            // 输出更新后的前k位和后k位列表
            System.out.println("Updated Genetic: " + updatedGenetic);
            System.out.println("Updated Particle: " + updatedParticle);
        }
    }
}
