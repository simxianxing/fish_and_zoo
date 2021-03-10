package ai.certifai.Day;

import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
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
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class imagec {

    static int epoch = 1;
    static int seed = 123;

    public static void main(String[] args) throws IOException {

        File trainFile = new ClassPathResource("/aitest/natural_images/seg_train/seg_train").getFile();
        File testFile = new ClassPathResource("/aitest/natural_images/seg_test/seg_test").getFile();
        FileSplit dataSplite = new FileSplit(trainFile);
        FileSplit dataSplite2 = new FileSplit(testFile);

        //CSVRecordReader csvdata = new CSVRecordReader(1, ',');
        ImageRecordReader trainimg = new ImageRecordReader(150, 150, 3);
        trainimg.initialize(dataSplite);

        ImageRecordReader testimg = new ImageRecordReader(150, 150, 3);
        testimg.initialize(dataSplite2);

        List<List<Writable>> traindata = new ArrayList<>();

        while(trainimg.hasNext()){
            traindata.add(trainimg.next());
        }

        List<List<Writable>> testdata = new ArrayList<>();

        while(testimg.hasNext()){
            testdata.add(testimg.next());
        }

        CollectionRecordReader trainRR = new CollectionRecordReader(traindata);
        DataSetIterator trainiter = new RecordReaderDataSetIterator(trainRR, traindata.size());

        CollectionRecordReader testRR = new CollectionRecordReader(testdata);
        DataSetIterator testiter = new RecordReaderDataSetIterator(testRR, testdata.size());

        DataSet testData = testiter.next();
        DataSet trainData = trainiter.next();

        ViewIterator trainIter = new ViewIterator(trainData, traindata.size());
        ViewIterator testIter = new ViewIterator(testData, testdata.size());



        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Nesterovs(0.01, Nesterovs.DEFAULT_NESTEROV_MOMENTUM))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(3)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(50).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(150, 150, 3)) // InputType.convolutional for normal image
                .backpropType(BackpropType.Standard)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        Evaluation eval;
        for(int i = 1; i <=  epoch; i++){
            model.fit(trainIter);
            eval = model.evaluate(trainIter);
            System.out.println("EPOCH: " + i + " Accuracy: " + eval.accuracy());
        }

        //  Evaluating the outcome of our trained model
        Evaluation evalTrain = model.evaluate(trainIter);
        Evaluation evalTest = model.evaluate(testIter);
        System.out.print("Train Data");
        System.out.println(evalTrain.stats());

        System.out.print("Test Data");
        System.out.print(evalTest.stats());


    }
}
