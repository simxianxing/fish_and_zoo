package ai.certifai.Day;

import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class morsecode {
    //private static final Logger log = org.slf4j.LoggerFactory.getLogger(EditAtBottleneckAndExtendModel.class);

    private static double trainPerc = 0.7;
    private static int width = 128;
    private static int height = 128;
    private static int channels = 3;
    private static int batchSize = 50;
    private static int numClass = 3;
    private static int numBox = 5;
    private static int epoch = 100;
    private static int seed = 123;
    private static double lr = 0.01;

    private static double[][] priorBoxes = {{1, 3}, {2.5, 6}, {3, 4}, {3.5, 8}, {4, 9}};


    private static String modelfilename = "D:\\TrainingLabs-main\\dl4j-labs\\src\\main\\java\\ai\\certifai\\Day\\morsecode.zip";

    public static void main(String[] args) throws IOException {
        File myFile = new ClassPathResource("finger").getFile();

        //Image Augmentation
        ImageTransform hFlip = new FlipImageTransform(1);

        //Image transform method, probability of images to get transform
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(new Pair<>(hFlip, 0.2));
        PipelineImageTransform tp = new PipelineImageTransform(pipeline, false);

        moesecodeiterator iterator = new moesecodeiterator();
        iterator.setup(myFile, trainPerc, width, height, channels, batchSize, numClass, tp);

        DataSetIterator trainIter = iterator.getTrain();
        DataSetIterator testIter = iterator.getTest();

        System.out.println(trainIter.getLabels());


//        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .weightInit(WeightInit.XAVIER)
//                .updater(new Adam(lr))
//                .list()
//                .layer(0, new ConvolutionLayer.Builder()
//                        .kernelSize(7, 7)
//                        .stride(1, 1)
//                        .nIn(channels)
//                        .activation(Activation.RELU)
//                        .nOut(64)
//                        .build())
//                .layer(1, new SubsamplingLayer.Builder()
//                        .kernelSize(3, 3)
//                        .stride(2, 2)
//                        .poolingType(SubsamplingLayer.PoolingType.MAX)
//                        .build())
//                .layer(2, new ConvolutionLayer.Builder()
//                        .kernelSize(3, 3)
//                        .stride(1, 1)
//                        .activation(Activation.RELU)
//                        .nOut(64)
//                        .build())
//                .layer(3, new SubsamplingLayer.Builder()
//                        .kernelSize(3, 3)
//                        .stride(2, 2)
//                        .poolingType(SubsamplingLayer.PoolingType.MAX)
//                        .build())
//                .layer(4, new ConvolutionLayer.Builder()
//                        .kernelSize(3, 3)
//                        .stride(1, 1)
//                        .activation(Activation.RELU)
//                        .nOut(128)
//                        .build())
//                .layer(5, new SubsamplingLayer.Builder()
//                        .kernelSize(3, 3)
//                        .stride(2, 2)
//                        .poolingType(SubsamplingLayer.PoolingType.MAX)
//                        .build())
//                .layer(6, new ConvolutionLayer.Builder()
//                        .kernelSize(3, 3)
//                        .stride(1, 1)
//                        .activation(Activation.RELU)
//                        .nOut(128)
//                        .build())
//                .layer(7, new SubsamplingLayer.Builder()
//                        .kernelSize(3, 3)
//                        .stride(2, 2)
//                        .poolingType(SubsamplingLayer.PoolingType.MAX)
//                        .build())
//                .layer(8, new ConvolutionLayer.Builder()
//                        .kernelSize(3, 3)
//                        .stride(1, 1)
//                        .activation(Activation.RELU)
//                        .nOut(256)
//                        .build())
//                .layer(9, new SubsamplingLayer.Builder()
//                        .kernelSize(2, 2)
//                        .stride(2, 2)
//                        .poolingType(SubsamplingLayer.PoolingType.MAX)
//                        .build())
//                .layer(10, new DenseLayer.Builder()
//                        .activation(Activation.RELU)
//                        .nOut(256)
//                        .build())
//                .layer(11, new DenseLayer.Builder()
//                        .activation(Activation.RELU)
//                        .nOut(128)
//                        .build())
//                .layer(12, new DenseLayer.Builder()
//                        .activation(Activation.RELU)
//                        .nOut(32)
//                        .build())
//                .layer(13, new OutputLayer.Builder()
//                        .activation(Activation.SOFTMAX)
//                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .nOut(numClass)
//                        .build())
//                .setInputType(InputType.convolutional(height, width, channels))
//                .build();
//
//        MultiLayerNetwork model = new MultiLayerNetwork(config);
//        model.init();

//        ZooModel zooModel = VGG16.builder().build();
//        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
//        //log.info(model.summary());
//        System.out.println(vgg16.summary());
//
//        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
//                .activation(Activation.LEAKYRELU)
//                .weightInit(WeightInit.XAVIER)
//                .updater(new Adam(lr))
//                .seed(seed)
//                .build();
//
//        ComputationGraph model = new TransferLearning.GraphBuilder(vgg16)
//                .fineTuneConfiguration(fineTuneConf)
//                .setFeatureExtractor("block5_pool") //"block5_pool" and below are frozen
//                .nOutReplace("fc2",1024, WeightInit.XAVIER) //modify nOut of the "fc2" vertex
//                .removeVertexAndConnections("predictions") //remove the final vertex and it's connections
//                .addLayer("fc3",new DenseLayer.Builder()
//                        .activation(Activation.RELU)
//                        .nIn(1024)
//                        .nOut(256)
//                        .build(),"fc2") //add in a new dense layer
//                .addLayer("newpredictions",new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .activation(Activation.SOFTMAX)
//                        .nIn(256)
//                        .nOut(numClass)
//                        .build(),"fc3") //add in a final output dense layer,
//                // configurations on a new layer here will be override the finetune confs.
//                // For eg. activation function will be softmax not RELU
//                .setOutputs("newpredictions") //since we removed the output vertex and it's connections we need to specify outputs for the graph
//                .build();

        Nd4j.getRandom().setSeed(seed);
        INDArray priors = Nd4j.create(priorBoxes);
        ComputationGraph pretrained = (ComputationGraph) YOLO2.builder().build().initPretrained();

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Adam.Builder().learningRate(lr).build())
                .l2(0.00001)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .build();

        ComputationGraph model = new TransferLearning.GraphBuilder(pretrained)
                                .fineTuneConfiguration(fineTuneConf)
                                .removeVertexKeepConnections("conv2d_23")
                                .removeVertexKeepConnections("outputs")
                                .addLayer("conv2d_23", new ConvolutionLayer.Builder(1, 1)
                                .nIn(1024)
                                .nOut(numBox * (5 + numClass))
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Same)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.IDENTITY)
                                .build(),
                        "leaky_re_lu_22")
                .addLayer("outputs",
                        new Yolo2OutputLayer.Builder()
                                .lambdaNoObj(0.5)
                                .lambdaCoord(5.0)
                                .boundingBoxPriors(priors.castTo(DataType.FLOAT))
                                .build(),
                        "conv2d_23")
                .setOutputs("outputs")
                .build();
        System.out.println(model.summary(InputType.convolutional(
                height,
                width,
                numClass)));




        //System.out.println(model.summary());

        model.setListeners(new ScoreIterationListener(1));

        Evaluation eval;
        for(int i = 1; i <=  epoch; i++){
            model.fit(trainIter);
            eval = model.evaluate(trainIter);
            System.out.println("EPOCH: " + i + " Accuracy: " + eval.accuracy());
        }

        ModelSerializer.writeModel(model, modelfilename, true);
        System.out.println("Model saved.");

        //  Evaluating the outcome of our trained model
        Evaluation evalTrain = model.evaluate(trainIter);
        Evaluation evalTest = model.evaluate(testIter);
        System.out.println("Train Evaluation: " + evalTrain.stats());
        System.out.println("Test Evaluation: " + evalTest.stats());

    }
}
