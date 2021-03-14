package ai.certifai.Day;


import org.datavec.image.transform.*;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class AvP {
    private static double trainPerc = 0.7;
    private static int width = 80;
    private static int height = 80;
    private static int channels = 3;
    private static int batchSize = 50;
    private static int numClass = 2;
    private static int epoch = 10;

    public static void main(String[] args) throws IOException {

        File myFile = new ClassPathResource("AvP").getFile();

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

        AvPIterator iterator = new AvPIterator();

        iterator.setup(myFile, trainPerc, width, height, channels, batchSize, numClass, tp);

        DataSetIterator trainIter = iterator.getTrain();
        DataSetIterator testIter = iterator.getTest();

        //model training

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nIn(channels)
                        .activation(Activation.RELU)
                        .nOut(24)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(20)
                        .build())
                .layer(new OutputLayer.Builder()
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClass)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        model.setListeners(new ScoreIterationListener(10));

        model.fit(trainIter, epoch);

        Evaluation eval = model.evaluate(trainIter);
        Evaluation evalTest = model.evaluate(testIter);

        System.out.println("Train Evaluation: " + eval.stats());
        System.out.println("Test Evaluation: " + evalTest.stats());







    }

}
