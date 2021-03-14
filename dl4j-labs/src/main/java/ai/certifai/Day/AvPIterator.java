package ai.certifai.Day;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class AvPIterator {
    //all things data loading
    private static Random rng = new Random();
    private static String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static BalancedPathFilter bPF = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
    private static InputSplit trainData, testData;
    private int width, height, channel, batchSize, numClass;
    private ImageTransform tp;

    public AvPIterator(){
        //constructor
    }

    public void setup(File file, double trainPerc, int widthA, int heightA, int channelA, int batchSizeA, int numClassA, ImageTransform tpA){

        width = widthA;
        height = heightA;
        channel = channelA;
        batchSize = batchSizeA;
        numClass = numClassA;
        FileSplit fsplit = new FileSplit(file);
        InputSplit[] allData = fsplit.sample(bPF, trainPerc, 1 - trainPerc);
        trainData = allData[0];
        testData = allData[1];
        tp = tpA;


        System.out.println(allData.length);


    }

    public DataSetIterator getTrain() throws IOException {
        return makeIterator(trainData, true);
    }

    public DataSetIterator getTest() throws IOException {
        return makeIterator(testData, false);
    }

    private DataSetIterator makeIterator(InputSplit data, boolean train) throws IOException {

        ImageRecordReader rr = new ImageRecordReader(height, width, channel, labelMaker);

        if(train){
            rr.initialize(data, tp);
        }
        else {
            rr.initialize(data);
        }


        DataSetIterator iter = new RecordReaderDataSetIterator(rr, batchSize, 1, numClass);

        DataNormalization scaler = new ImagePreProcessingScaler();
        iter.setPreProcessor(scaler);

        return iter;
    }

}
