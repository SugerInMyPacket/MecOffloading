package utils;

import java.io.*;
import java.util.Random;

public class WriteCSV {

    public static void main(String[] args) throws Exception {
        // writeTest();
        writeTask();
    }

    public static void writeTest() throws Exception {
        //第一步：设置输出的文件路径
        //如果该目录下不存在该文件，则文件会被创建到指定目录下。如果该目录有同名文件，那么该文件将被覆盖。
        File writeFile = new File("../data/test.csv");

        try {
            //第二步：通过BufferedReader类创建一个使用默认大小输出缓冲区的缓冲字符输出流
            BufferedWriter writeText = new BufferedWriter(new FileWriter(writeFile));

            //第三步：将文档的下一行数据赋值给lineData，并判断是否为空，若不为空则输出
            for (int i = 1; i <= 10; i++) {
                writeText.newLine();    //换行
                //调用write的方法将字符串写到流中
                writeText.write("新用户" + i + ",男," + (18 + i));
            }

            //使用缓冲区的刷新方法将数据刷到目的地中
            writeText.flush();
            //关闭缓冲区，缓冲区没有调用系统底层资源，真正调用底层资源的是FileWriter对象，缓冲区仅仅是一个提高效率的作用
            //因此，此处的close()方法关闭的是被缓存的流对象
            writeText.close();
        } catch (FileNotFoundException e) {
            System.out.println("没有找到指定文件");
        } catch (IOException e) {
            System.out.println("文件读写出错");
        }
    }

    public static void writeTask() throws Exception {
        //第一步：设置输出的文件路径
        //如果该目录下不存在该文件，则文件会被创建到指定目录下。如果该目录有同名文件，那么该文件将被覆盖。
        File writeFile = new File("../data/task01.csv");

        try {
            //第二步：通过BufferedReader类创建一个使用默认大小输出缓冲区的缓冲字符输出流
            BufferedWriter writeText = new BufferedWriter(new FileWriter(writeFile));

            //第三步：将文档的下一行数据赋值给lineData，并判断是否为空，若不为空则输出
            for (int i = 0; i < 1000; i++) {
                writeText.newLine();    //换行
                //调用write的方法将字符串写到流中
                writeText.write(i + ",");  // task_id
                writeText.write((new Random().nextInt(5) + 1) + ",");  // size   [1, 5]  ===> MB/10^6
                writeText.write((double) (new Random().nextInt(20) + 1) / 100.0 + ",");  // rate  [0.01, 0.2]
                // CPU cycles / bit    [10, 100, 1000]
                int cpu_cycles_bit = 0;
                int kind = i % 3 + 1;
                if(kind == 1) {
                    cpu_cycles_bit = 10;
                } else if (kind == 2) {
                    cpu_cycles_bit = 100;
                } else if (kind == 3) {
                    cpu_cycles_bit = 1000;
                }
                writeText.write(cpu_cycles_bit + ",");  // c
                // 当前时刻记录为0, deadline传入一个整数
                writeText.write((new Random().nextInt(1000) + 100) + ",");  // deadline
                writeText.write((new Random().nextInt(5) + 1) + ",");  // factor  [1, 5]
                writeText.write(kind + ",");  // I
                writeText.write((new Random().nextInt(10) + 1) + ",");  // prior  [1, 10]
                writeText.write((new Random().nextInt(50) + 1) + "");  // vehicle_id [1, 50]
            }

            //使用缓冲区的刷新方法将数据刷到目的地中
            writeText.flush();
            //关闭缓冲区，缓冲区没有调用系统底层资源，真正调用底层资源的是FileWriter对象，缓冲区仅仅是一个提高效率的作用
            //因此，此处的close()方法关闭的是被缓存的流对象
            writeText.close();
            System.out.println("------- 写入task完成！！！ -------");
        } catch (FileNotFoundException e) {
            System.out.println("没有找到指定文件");
        } catch (IOException e) {
            System.out.println("文件读写出错");
        }
    }
}
