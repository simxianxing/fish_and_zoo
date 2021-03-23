package ai.certifai.Day;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Random;

public class moesecodeiterator {

    private static final Random rng = new Random();
    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static final ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static final BalancedPathFilter bPF = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
    private static InputSplit trainData, testData;
    private int width, height, channel, batchSize, numClass;
    private ImageTransform tp;
    private Path dir;
    public static final int gridWidth = 13;
    public static final int gridHeight = 13;

    public moesecodeiterator(){
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
        dir = file.toPath();

        System.out.println(trainData.length());
        System.out.println(testData.length());

    }

    public RecordReaderDataSetIterator getTrain(int batchSize) throws Exception {
        return makeIterator(trainData, dir, batchSize);
    }

    public RecordReaderDataSetIterator getTest(int batchSize) throws Exception {
        return makeIterator(testData, dir, batchSize);
    }

    private RecordReaderDataSetIterator makeIterator(InputSplit data, Path dir, int batchSize) throws IOException {

        ObjectDetectionRecordReader recordReader = new ObjectDetectionRecordReader(height, width, channel, gridHeight, gridWidth, new VocLabelProvider(dir.toString()));

        recordReader.initialize(data);

        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, 1, true);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        iter.setPreProcessor(scaler);

        System.out.println(iter.toString());

        return iter;

    }

}
