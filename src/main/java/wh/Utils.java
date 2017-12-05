package wh;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

/**
 * Created by WangHong on 2017/12/5.
 */
public class Utils {
    public static Random rand = new Random(System.nanoTime());
    /**
     * 改进的k-modes聚类
     * @param ins 标签样本
     * @param k 类簇数
     * @param inum 最大迭代次数
     * @param maxc 可接受的最大误差
     * @return
     * @throws IllegalArgumentException
     */
    public static double[][] kModesCossimil(double[][] ins,int k,int inum,double maxc) throws Exception{
        //数据检查
        if(ins.length<k || k<1) throw new IllegalArgumentException("聚类类簇数大于距离数");
        if(inum<1 && maxc<0.1) throw new IllegalArgumentException("聚类条件输入错误");
        //将标签中的0替换为-1，使数据归一化及可以正确计算余弦相似度
        for (int i = 0; i < ins.length; i++) {
            for (int j = 0; j < ins[i].length; j++) {
                if(ins[i][j]==0)
                    ins[i][j]=-1;
            }
        }
        //随机初始化K个不同的蔟中心点
        int i = 0;
        double[][] centers = new double[k][];

        //按照K-means++的方式初始化K个蔟中心点
        HashSet<Integer> cIdxMap = new HashSet<>(k);
        int idx1 = rand.nextInt(ins.length);
        centers[0]=ins[idx1];
        cIdxMap.add(idx1);
        for (int j = 1; j < k; j++) {
            double[] D = new double[ins.length];
            double sum = 0;
            for (int l = 0; l < ins.length; l++) {
                if(cIdxMap.contains(l)){
                    D[l]=0;
                }else {
                    double mix = Double.MAX_VALUE;
                    for (int m = 0; m < j; m++) {
                        double i2cDis = expAbsCosSim(centers[m],ins[l])[0];
                        mix = mix>i2cDis?i2cDis:mix;
                    }
                    D[l]=mix;
                    sum+=mix;
                }
            }
            //根据概率选择距离较大的点作为中心
            sum *= rand.nextDouble();
            int l = -1;
            while (sum>0){
                l++;
                sum-=D[l];

            }
            if(cIdxMap.contains(l)) throw new Exception("初始化聚类中心出现重复");
            else {
                centers[j] = ins[l];
                cIdxMap.add(l);
            }
        }
        //记录聚类中心改变量
        double changeNum = Double.MAX_VALUE;
        //记录迭代次数
        i = 0;
        //记录每个蔟信息
        ArrayList<ArrayList<CluSampInfo>> clusters = new ArrayList<>(k);
        while (i<inum && changeNum>maxc){
            //聚类中心改变量重置
            changeNum = 0.0;
            //每次重新初始化类簇信息
            clusters.clear();
            for (int j = 0; j < k; j++) {
                clusters.add(new ArrayList<>(ins.length/k + 1));
            }
            //对于每个样本，计算和所有蔟中心的距离，将其归为距离最近的类簇
            for (int j = 0; j < ins.length; j++) {
                double minDis = Double.MAX_VALUE;
                int minCenIdx = -1;
                double minCos = -1;
                for (int l = 0; l < centers.length; l++) {
                    double[] tmpDisCos = expAbsCosSim(centers[l],ins[j]);
                    if(tmpDisCos[0]<=minDis){
                        minDis = tmpDisCos[0];
                        minCenIdx = l;
                        minCos = tmpDisCos[1];
                    }
                }
                clusters.get(minCenIdx).add(new CluSampInfo(j,minDis,minCos>0));

            }
            //根据新类簇重新计算每个类簇的中心，并记录总体中心距离改变
            for (int j = 0; j < clusters.size(); j++) {
                ArrayList<CluSampInfo> tmp = clusters.get(j);
                //用于重新计算中心
                double[] newCenter = new double[ins[0].length];
                for (int l = 0; l < tmp.size(); l++) {
                    CluSampInfo tmpInfo = tmp.get(l);
                    double [] aSample = ins[tmpInfo.getIndex()];
                    //区分正负相关，负相关向量映射为正相关
                    if(tmpInfo.isPosCor()){
                        for (int m = 0; m < newCenter.length; m++) {
                            newCenter[m] += aSample[m];
                        }
                    }else{
                        for (int m = 0; m < newCenter.length; m++) {
                            newCenter[m] -= aSample[m];
                        }
                    }
                }
                for (int l = 0; l < newCenter.length; l++) {
                    //按照众数
                    if(newCenter[l]>=0)
                        newCenter[l] = 1;
                    else
                        newCenter[l] = -1;
                    //按照平均值
                    /*newCenter[l] = newCenter[l]*1.0/tmp.size();*/
                }
                //累加距离改变量
                changeNum += expAbsCosSim(centers[j],newCenter)[0];
                //更新聚类中心
                centers[j] = newCenter;
            }
            i++;
        }
        return centers;
    }

    /**
     * 基于余弦相似度的改进距离计算公式
     * @param v1
     * @param v2
     * @return double[] -> 2元素 距离值，余弦相似度
     */
    public static double[] expAbsCosSim(double[] v1, double[] v2) throws IllegalArgumentException{
        if(v1.length!=v2.length || v1.length<1) throw new IllegalArgumentException("向量计算距离维度不匹配或长度小于1");
        double cosup = 0.0;
        double cosdownv1 = 0.0;
        double cosdownv2 = 0.0;
        for (int i = 0; i < v1.length; i++) {
            cosup += v1[i]*v2[i];
            cosdownv1 += Math.pow(v1[i],2);
            cosdownv2 += Math.pow(v2[i],2);
        }
        double cos = cosup/(Math.sqrt(cosdownv1)*Math.sqrt(cosdownv2));
        if(cos==0) cos = Double.MIN_VALUE;
        double[] res = {-1.0*Math.log(Math.abs(cos)),cos};
        return res;
    }
}

//保存聚类类簇中的样本信息
class CluSampInfo{
    //样本索引
    private int index;
    //到类中心的距离
    private double dis;
    //正相关为true,负相关为false
    private boolean isPosCor;

    public CluSampInfo(int index, double dis, boolean isPosCor) {
        this.index = index;
        this.dis = dis;
        this.isPosCor = isPosCor;
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public double getDis() {
        return dis;
    }

    public void setDis(double dis) {
        this.dis = dis;
    }

    public boolean isPosCor() {
        return isPosCor;
    }

    public void setPosCor(boolean posCor) {
        isPosCor = posCor;
    }

    @Override
    public String toString() {
        return "wh.CluSampInfo{" +
                "index=" + index +
                ", dis=" + dis +
                ", isPosCor=" + isPosCor +
                '}';
    }
}
