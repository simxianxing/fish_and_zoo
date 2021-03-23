package ai.certifai.Day;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class imagec {

    private static Random rng = new Random();
    private static String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static BalancedPathFilter bPF = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
    private static double trainPerc = 0.7;
    private static int height = 150;
    private static int width = 150;
    private static int channel = 3;
    private static int epoch = 100;
    private static int seed = 123;
    private static int batchSize = 1000;
    private static int numClass = 6;
    private static double lr = 0.001;

    public static void main(String[] args) throws IOException {

        File file = new ClassPathResource("/aitest/natural_images/seg_train/seg_train").getFile();

        //image Augmentation
        ImageTransform hFlip = new FlipImageTransform(1);
        ImageTransform rotate = new RotateImageTransform(15);
        ImageTransform crop = new CropImageTransform(5);

        //Image transform method, probability of images to get transform
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(hFlip, 0.2),
                new Pair<>(rotate, 0.3),
                new Pair<>(crop, 0.2)
        );

        PipelineImageTransform tp = new PipelineImageTransform(pipeline, false);


        FileSplit fsplit = new FileSplit(file);

        InputSplit[] allData = fsplit.sample(bPF, trainPerc, 1 - trainPerc);
        InputSplit trainData = allData[0];
        InputSplit testData = allData[1];


        ImageRecordReader trainimg = new ImageRecordReader(height, width, channel, labelMaker);
        trainimg.initialize(trainData, tp);

        ImageRecordReader testimg = new ImageRecordReader(height, width, channel, labelMaker);
        testimg.initialize(testData);

        System.out.println(testimg.getLabels());

        DataSetIterator trainiter = new RecordReaderDataSetIterator(trainimg, (int) (0.7*batchSize), 1, numClass);
        DataSetIterator testiter = new RecordReaderDataSetIterator(testimg, (int) (0.3*batchSize), 1, numClass);

        DataNormalization scaler = new ImagePreProcessingScaler();
        trainiter.setPreProcessor(scaler);
        testiter.setPreProcessor(scaler);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(lr))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new ConvolutionLayer.Builder()
                        .kernelSize(5, 5)
                        .nIn(channel)
                        .stride(1, 1)
                        .nOut(16)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nOut(16)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build())
                .layer(4, new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build())
                .layer(6, new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build())
                .layer(8, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(32)
                        .build())
                .layer(9, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(32)
                        .build())
                .layer(10, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(16)
                        .build())
                .layer(11, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(6)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(150, 150, 3)) // InputType.convolutional for normal image
                .backpropType(BackpropType.Standard)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);

        model.init();
        System.out.println(model.summary());
        model.setListeners(new ScoreIterationListener(1));

        Evaluation eval;
        for(int i = 1; i <=  epoch; i++){
            model.fit(trainiter);
            eval = model.evaluate(trainiter);
            System.out.println("EPOCH: " + i + " Accuracy: " + eval.accuracy());
        }

        //  Evaluating the outcome of our trained model
        Evaluation evalTrain = model.evaluate(trainiter);
        Evaluation evalTest = model.evaluate(testiter);
        System.out.print("Train Data");
        System.out.println(evalTrain.stats());

        System.out.print("Test Data");
        System.out.print(evalTest.stats());

        ModelSerializer.writeModel(model, new File("D:/TrainingLabs-main/dl4j-labs/src/main/java/ai/certifai/Day/imagec.zip"), true);


    }

}
