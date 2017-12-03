package mulan.classifier.transformation;


import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.clusterers.SimpleKMeans;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.math.BigDecimal;
import java.util.*;

/**
 * 局部标签关联分类器链算法
 * Created by WangHong on 2017/11/17.
 */
public class ClusterLocalClassifierChains extends TransformationBasedMultiLabelLearner{
    /**
     * 局部分类器链数
     */
    protected int numOfLocalModels;
    /**
     * 保存局部分类器链的数组
     */
    protected LocalClassifieChain[] lccs;
    /**
     * 保存局部分类器链顺序的数组
     */
    protected int[][] lcIdx;
    /**
     * 随机数生成器
     */
    protected Random rand;
    /**
     * 是否打印调试信息
     */
    protected boolean debug;

    /**
     * 聚类最大迭代次数
     */
    protected int iterNum;

    /**
     * 聚类可接受最大中心改变量
     */
    protected double maxChange;

    /**
     * 采用的内部算法 1 互信息 2 k-means 3 改进的余弦相似度kmodes
     */
    int flag;

    /**
     * Default constructor
     */
    public ClusterLocalClassifierChains() {
        this(new J48(), 2, 10, 0.0, true, 1);
    }

    /**
     * Creates a new object
     *
     * @param classifier 基分类器
     * @param aNumOfLocalModels 局部分类器链数
     */
    public ClusterLocalClassifierChains(Classifier classifier, int aNumOfLocalModels, int iterNum, double maxChange, boolean isDebug, int flag) {
        super(classifier);
        this.numOfLocalModels = aNumOfLocalModels;
        this.lccs = new LocalClassifieChain[aNumOfLocalModels];
        this.rand = new Random(System.nanoTime());
        this.iterNum = iterNum;
        this.maxChange = maxChange;
        this.debug = isDebug;
        this.flag = flag;
    }

    /**
     * 基于余弦相似度的改进距离计算公式
     * @param v1
     * @param v2
     * @return double[] -> 2元素 距离值，余弦相似度
     */
    private double[] expAbsCosSim(double[] v1, double[] v2) throws IllegalArgumentException{
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


    /**
     * 欧式距离
     * @param v1
     * @param v2
     * @return
     * @throws IllegalArgumentException
     */
    private double euclDis(double[] v1,double[] v2) throws IllegalArgumentException{
        if(v1.length!=v2.length) throw new IllegalArgumentException("向量计算距离维度不匹配");
        double tmp = 0.0;
        for (int i = 0; i < v1.length; i++) {
            tmp+=Math.pow(v1[i]-v2[i],2);
        }
        return Math.sqrt(tmp);
    }

    /***
     * 计算两个向量之间的互信息
     * @param a 向量a
     * @param b 向量b
     * @return 保留3位小数的互信息值
     */
    private double getMI(double[] a,double[] b) throws IllegalArgumentException{
        if(a.length!=b.length)
            throw new IllegalArgumentException("向量计算距离维度不匹配");

        HashMap map = new HashMap();
        HashMap xmap = new HashMap();
        HashMap ymap = new HashMap();
        for (int i = 0; i <a.length ; i++) {
            String xyKey = String.format("%s%s",(int)a[i],(int)b[i]);
            String xKey = String.format("%s",(int)a[i]);
            String yKey = String.format("%s",(int)b[i]);
            if(map.containsKey(xyKey)){
                map.put(xyKey,Integer.parseInt(map.get(xyKey).toString())+1);
            }
            else map.put(xyKey,1);
            if(xmap.containsKey(xKey)){
                xmap.put(xKey,Integer.parseInt(xmap.get(xKey).toString())+1);
            }
            else xmap.put(xKey,1);
            if(ymap.containsKey(yKey)){
                ymap.put(yKey,Integer.parseInt(ymap.get(yKey).toString())+1);
            }
            else ymap.put(yKey,1);
        }
        double sum = 0;
        for (Object o:map.keySet()) {
            String key = o.toString();
            String x = key.charAt(0)+"";
            String y = key.charAt(1)+"";
            int valueXY = Integer.parseInt(map.get(o).toString());
            int valueX = Integer.parseInt(xmap.get(x).toString());
            int valueY = Integer.parseInt(ymap.get(y).toString());
            double pxy = valueXY/(a.length*1.0);
            double px = valueX/(a.length*1.0);
            double py = valueY/(a.length*1.0);
            sum+=pxy*Math.log(pxy/(px*py));
        }
        return new BigDecimal(sum).setScale(3,BigDecimal.ROUND_HALF_UP).doubleValue();
    }

    //整形数组变浮点型数组
    private double[] i2darr(int[] arr){
        double[] newarr=new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            newarr[i] = arr[i];
        }
        return newarr;
    }

    //返回数组中有多少个-1
    private int getN1Num(double[] arr){
        int res = 0;
        for (int i = 0; i < arr.length; i++) {
            if(arr[i]==-1.0) res++;
        }
        return res;
    }


    /**
     *
     * @param ins
     * @return
     * @throws Exception
     */
    private int[] MIMax(int[][] ins, int[] lidxs) throws Exception {
        if(lidxs.length==1) return lidxs;
        //数据检查
        HashMap<Integer,double[]> miMap = new HashMap<>(lidxs.length);
        for (int i = 0; i < lidxs.length; i++) {
            double[] tmp = new double[lidxs.length];
            for (int j = 0; j < lidxs.length; j++) {
                if(i==j) tmp[j]=-1;
                else tmp[j] = getMI(i2darr(ins[lidxs[i]]),i2darr(ins[lidxs[j]]));
            }
            miMap.put(i,tmp);
        }
        for (int i = 0; i < miMap.size(); i++) {
            System.out.println(i + " -> " + Arrays.toString(miMap.get(i)));
        }

        //找出互信息最小的横纵坐标
        int minx = -1;
        int miny = -1;
        double minv = Double.MAX_VALUE;
        for (int i = 0; i < lidxs.length; i++) {
            for (int j = i+1; j < lidxs.length; j++) {
                double[] tmp = miMap.get(i);
                if(minv>tmp[j]){
                    minv = tmp[j];
                    minx = i;
                    miny = j;
                }
            }
        }

        int nextIdx = rand.nextInt(lidxs.length)%2==0?minx:miny;
        int[] res = new int[lidxs.length];
        res[0]=nextIdx;
        int i = 1;
        while (miMap.size()>1){
            double[] tmp = miMap.get(nextIdx);
            miMap.remove(nextIdx);
            int minIdx = -1;
            double minValue = -0.1;
            for (int j = 0; j < tmp.length; j++) {
                if(minValue<tmp[j]){
                    minValue=tmp[j];
                    minIdx = j;
                }
            }
            nextIdx = minIdx;
            tmp[minIdx] = -0.1;
            while (!miMap.containsKey(nextIdx)){
                minIdx = -1;
                minValue = -0.1;
                for (int j = 0; j < tmp.length; j++) {
                    if(minValue<tmp[j]){
                        minValue=tmp[j];
                        minIdx = j;
                    }
                }
                nextIdx = minIdx;
                tmp[minIdx] = -0.1;
            }
            res[i] =nextIdx;
            i++;
        }
        System.out.println(Arrays.toString(res));
        for (int j = 0; j < res.length; j++) {
            res[j] = lidxs[res[j]];
        }
        return res;
    }


    /**
     *
     * @param ins
     * @return
     * @throws Exception
     */
    private int[] MIMix(int[][] ins, int[] lidxs) throws Exception {
        if(lidxs.length==1) return lidxs;

        //数据检查
        HashMap<Integer,double[]> miMap = new HashMap<>(lidxs.length);
        for (int i = 0; i < lidxs.length; i++) {
            double[] tmp = new double[lidxs.length];
            for (int j = 0; j < lidxs.length; j++) {
                if(i==j) tmp[j]=Double.MAX_VALUE;
                else tmp[j] = getMI(i2darr(ins[lidxs[i]]),i2darr(ins[lidxs[j]]));
            }
            miMap.put(i,tmp);
        }
        if(debug){
        for (int i = 0; i < miMap.size(); i++) {
            System.out.println(i + " -> " + Arrays.toString(miMap.get(i)));
        }}

        //找出互信息最小的横纵坐标
        int minx = -1;
        int miny = -1;
        double minv = Double.MAX_VALUE;
        for (int i = 0; i < lidxs.length; i++) {
            for (int j = i+1; j < lidxs.length; j++) {
                double[] tmp = miMap.get(i);
                if(minv>tmp[j]){
                    minv = tmp[j];
                    minx = i;
                    miny = j;
                }
            }
        }

        int nextIdx = rand.nextInt(lidxs.length)%2==0?minx:miny;
        int[] res = new int[lidxs.length];
        res[0]=nextIdx;
        int i = 1;
        while (miMap.size()>1){
            double[] tmp = miMap.get(nextIdx);
            miMap.remove(nextIdx);
            int minIdx = -1;
            double minValue = Double.MAX_VALUE;
            for (int j = 0; j < tmp.length; j++) {
                if(minValue>tmp[j]){
                    minValue=tmp[j];
                    minIdx = j;
                }
            }
            nextIdx = minIdx;
            tmp[minIdx] = Double.MAX_VALUE;
            while (!miMap.containsKey(nextIdx)){
                minIdx = -1;
                minValue = Double.MAX_VALUE;
                for (int j = 0; j < tmp.length; j++) {
                    if(minValue>tmp[j]){
                        minValue=tmp[j];
                        minIdx = j;
                    }
                }
                nextIdx = minIdx;
                tmp[minIdx] = Double.MAX_VALUE;
            }
            res[i] =nextIdx;
            i++;
        }
        if(debug){
        System.out.println(Arrays.toString(res));}
        for (int j = 0; j < res.length; j++) {
            res[j] = lidxs[res[j]];
        }
        return res;
    }

    private int[][] EMI(int[][] ins,int k,int inum,double maxc) throws Exception{
        int[][] cluster = Kmeans(ins,k,inum,maxc);
        for (int i = 0; i < cluster.length; i++) {
            Arrays.sort(cluster[i]);
        }
        for (int i = 0; i < cluster.length; i++) {
            cluster[i] = MIMix(ins,cluster[i]);
        }
        if(debug){
        for (int i = 0; i < cluster.length; i++) {
            System.out.println(Arrays.toString(cluster[i]));
        }}
        return cluster;
    }

    /**
     * 原始kmeans
     * @param ins 标签样本
     * @param k 类簇数
     * @param inum 最大迭代次数
     * @param maxc 可接受的最大误差
     * @return
     * @throws Exception
     */
    private int[][] Kmeans(int[][] ins,int k,int inum,double maxc) throws Exception{
        //数据检查
        if(ins.length<k || k<1) throw new IllegalArgumentException("聚类类簇数大于距离数");
        if(inum<1 && maxc<0.1) throw new IllegalArgumentException("聚类条件输入错误");
        //构建instances
        ArrayList<Attribute> atts = new ArrayList<>(ins[0].length);
        for (int i = 0; i < ins[0].length; i++) {
                atts.add(new Attribute("f"+i));
        }
        Instances cluIns = new Instances("cluins",atts,0);
        for (int i = 0; i < ins.length; i++) {
            cluIns.add(new DenseInstance(1.0,i2darr(ins[i])));
        }
        //聚类
        SimpleKMeans kMeans = new SimpleKMeans();
        kMeans.setMaxIterations(inum);
        kMeans.setNumClusters(k);
        kMeans.setInitializeUsingKMeansPlusPlusMethod(true);
        kMeans.buildClusterer(cluIns);
        Instances clu_cen = kMeans.getClusterCentroids();
        double[][] centers = new double[clu_cen.size()][];
        for (int i = 0; i < clu_cen.size(); i++) {
            centers[i] = clu_cen.get(i).toDoubleArray();
        }
        ArrayList<ArrayList<CluSampInfo>> clusters = new ArrayList<>(centers.length);
        for (int i = 0; i < centers.length; i++) {
            clusters.add(new ArrayList<>());
        }
        for (int i = 0; i < ins.length; i++) {
            int mixIdx = -1;
            double mixVlu = Double.MAX_VALUE;
            for (int j = 0; j < centers.length; j++) {
                double tmp = euclDis(centers[j],i2darr(ins[i]));
                if(tmp<mixVlu){
                    mixIdx = j;
                    mixVlu = tmp;
                }
            }
            clusters.get(mixIdx).add(new CluSampInfo(i,mixVlu,true));
        }
        //排序
        int[][] res = new int[clusters.size()][];
        for (int j = 0; j < clusters.size(); j++) {
            clusters.get(j).sort(new SortByDisDsc());
        }
        for (int j = 0; j < clusters.size(); j++) {
            ArrayList<CluSampInfo> tmp = clusters.get(j);
            int[] tmparr = new int[tmp.size()];
            for (int l = 0; l < tmparr.length; l++) {
                tmparr[l] = tmp.get(l).getIndex();
            }
            res[j] = tmparr;
        }
        if(debug){
            for (int j = 0; j < res.length; j++) {
                System.out.println(Arrays.toString(res[j]));
            }
        }
        return res;
    }

    /**
     * 改进的k-modes聚类
     * @param ins 标签样本
     * @param k 类簇数
     * @param inum 最大迭代次数
     * @param maxc 可接受的最大误差
     * @return
     * @throws IllegalArgumentException
     */
    private int[][] kModesCossimil(int[][] ins,int k,int inum,double maxc) throws Exception{
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
        /*HashSet<Integer> initCenterIdx = new HashSet<>(k);
        while (initCenterIdx.size()<k){
            initCenterIdx.add(rand.nextInt(ins.length));
        }
        Iterator<Integer> initCenterItor = initCenterIdx.iterator();
        while (initCenterItor.hasNext()){
            centers[i] = ins[initCenterItor.next()];
            i++;
        }*/

        //按照距离最远原则初始化k个蔟中心点
        /*int idx1 = rand.nextInt(ins.length);
        centers[0] = i2darr(ins[idx1]);
        for (int j = 1; j < k; j++) {
            double maxDis = -1;
            int maxIdx = -1;
            for (int l = 0; l < ins.length; l++) {
                double mix = Double.MAX_VALUE;
                for (int m = 0; m < j; m++) {
                    double i2cDis = expAbsCosSim(centers[m],i2darr(ins[l]))[0];
                    mix = mix>i2cDis?i2cDis:mix;
                }
                if(maxDis<mix){
                    maxDis = mix;
                    maxIdx = l;
                }
            }

            centers[j] = i2darr(ins[maxIdx]);
        }*/

        //按照K-means++的方式初始化K个蔟中心点
        HashSet<Integer> cIdxMap = new HashSet<>(k);
        int idx1 = rand.nextInt(ins.length);
        centers[0]=i2darr(ins[idx1]);
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
                        double i2cDis = expAbsCosSim(centers[m],i2darr(ins[l]))[0];
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
                centers[j] = i2darr(ins[l]);
                cIdxMap.add(l);
            }
        }
        for (int j = 0; j < centers.length; j++) {
            System.out.println(Arrays.toString(centers[j]));
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
                    double[] tmpDisCos = expAbsCosSim(centers[l],i2darr(ins[j]));
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
                    int [] aSample = ins[tmpInfo.getIndex()];
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
        int[][] res = new int[k][];
        for (int j = 0; j < clusters.size(); j++) {
            clusters.get(j).sort(new SortByDisDsc());
        }
        for (int j = 0; j < clusters.size(); j++) {
            ArrayList<CluSampInfo> tmp = clusters.get(j);
            int[] tmparr = new int[tmp.size()];
            for (int l = 0; l < tmparr.length; l++) {
                tmparr[l] = tmp.get(l).getIndex();
            }
            res[j] = tmparr;
        }
        return res;
    }

    private void computeLcIdx(Instances dataSet) throws Exception {
        //只保留标签列
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(labelIndices);
        remove.setInvertSelection(true);
        remove.setInputFormat(dataSet);
        Instances labels = Filter.useFilter(dataSet, remove);
        //标签数据转置
        int[][] labelIns = new int[labelIndices.length][labels.size()];
        for (int i = 0; i < labels.size(); i++) {
            String[] tmp = new DenseInstance(labels.get(i)).toString().split(",");
            for (int j = 0; j < tmp.length; j++) {
                labelIns[j][i] = Integer.valueOf(tmp[j]);
            }
        }
        //聚类，并得到lcIdx
        if(flag==1){
            this.lcIdx=EMI(labelIns,this.numOfLocalModels,this.iterNum,this.maxChange);
        }else if(flag==2){
            this.lcIdx=Kmeans(labelIns,this.numOfLocalModels,this.iterNum,this.maxChange);
        }else if(flag==3){
            this.lcIdx=kModesCossimil(labelIns,this.numOfLocalModels,this.iterNum,this.maxChange);
        }else{

        }


    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        Instances dataSet = new Instances(trainingSet.getDataSet());
        computeLcIdx(dataSet);
        //根据lcIdx训练k个CC链
        for (int i = 0; i < this.lcIdx.length; i++) {
            this.lccs[i] = new LocalClassifieChain(baseClassifier, this.lcIdx[i]);
            System.out.println(i+" baseClassifier start build");
            this.lccs[i].build(trainingSet);
            System.out.println(i+" baseClassifier has built");
        }
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        boolean[] bipartition = new boolean[numLabels];
        double[] confidences = new double[numLabels];
        for (int i = 0; i < this.lcIdx.length; i++) {
            MultiLabelOutput localChainO = this.lccs[i].makePrediction(instance);
            boolean[] bip = localChainO.getBipartition();
            double[] conf = localChainO.getConfidences();
            for (int j = 0; j < this.lcIdx[i].length; j++) {
                bipartition[this.lcIdx[i][j]]=bip[j];
                confidences[this.lcIdx[i][j]]=conf[j];
            }
        }
        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
        return mlo;
    }

    public static void main(String[] args) throws Exception{
        /*String path = "./data/testData/";
        MultiLabelLearnerBase learner = new ClusterLocalClassifierChains(new J48(), 1, 100, 0.0, true, 3);

        String trainDatasetPath = path + "CAL500-train.arff";
        String testDatasetPath = path + "CAL500-test.arff";
        String xmlLabelsDefFilePath = path + "CAL500.xml";
        MultiLabelInstances trainDataSet = new MultiLabelInstances(trainDatasetPath, xmlLabelsDefFilePath);
        MultiLabelInstances testDataSet = new MultiLabelInstances(testDatasetPath, xmlLabelsDefFilePath);

        learner.build(trainDataSet);*/
        //System.out.println(learner.makePrediction(testDataSet.getDataSet().firstInstance()));
    }
}

//按照距离从大到小进行排序
class SortByDisDsc implements Comparator<CluSampInfo>{
    @Override
    public int compare(CluSampInfo o1, CluSampInfo o2) {
        return o1.getDis()>=o2.getDis()?-1:1;
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
        return "CluSampInfo{" +
                "index=" + index +
                ", dis=" + dis +
                ", isPosCor=" + isPosCor +
                '}';
    }
}
