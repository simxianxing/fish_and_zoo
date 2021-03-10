package ai.certifai.Day;

import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class toyotaprice {

    static int seed = 123;
    static double lr = 0.01;
    static int epoch = 1;
    static int batchSize = 786;

    public static void main(String[] args) throws IOException, InterruptedException {
        File data = new ClassPathResource("archive/toyota.csv").getFile();
        FileSplit dataSplite = new FileSplit(data);
        CSVRecordReader csvdata = new CSVRecordReader(1, ',');
        csvdata.initialize(dataSplite);

        Schema sc =new Schema.Builder()
                .addColumnCategorical("model", Arrays.asList(" GT86", " Corolla", " RAV4", " Yaris", " Auris", " Aygo", " C-HR", " Prius", " Avensis", " Verso", " Hilux", " PROACE VERSO", " Land Cruiser", " Supra", " Camry", " Verso-S", " IQ", " Urban Cruiser"))
                .addColumnInteger("year")
                .addColumnInteger("price")
                .addColumnCategorical("transmission", Arrays.asList("Automatic", "Semi-Auto", "Manual", "Other"))
                .addColumnInteger("mileage")
                .addColumnCategorical("fuelType", Arrays.asList("Diesel", "Petrol", "Hybrid", "Other"))
                .addColumnInteger("tax")
                .addColumnDouble("mpg")
                .addColumnDouble("engineSize")
                .build();

        TransformProcess tp = new TransformProcess.Builder(sc)
                .categoricalToInteger("model")
                .categoricalToInteger("transmission")
                .categoricalToInteger("fuelType")
                .build();

        System.out.println(tp.getFinalSchema());

        List<List<Writable>> alldata = new ArrayList<>();

        while(csvdata.hasNext()){
            alldata.add(csvdata.next());
        }

        List<List<Writable>> transformedData = LocalTransformExecutor.execute(alldata, tp);

        for (List<Writable> transformedDatum : transformedData) {
            System.out.println(transformedDatum);
        }

        CollectionRecordReader crr = new CollectionRecordReader(transformedData);
        DataSetIterator dataIter = new RecordReaderDataSetIterator(crr,transformedData.size(),2,2, true);

        DataSet allData = dataIter.next();
        allData.shuffle();

        System.out.println(allData);


        SplitTestAndTrain testTrainSplit = allData.splitTestAndTrain(0.7);

        DataSet trainingSet = testTrainSplit.getTrain();
        DataSet testSet = testTrainSplit.getTest();

        System.out.println("Training vector : ");
        System.out.println(Arrays.toString(trainingSet.getFeatures().shape()));
        System.out.println("Test vector : ");
        System.out.println(Arrays.toString(testSet.getFeatures().shape()));

        DataNormalization normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(trainingSet);
        normalizer.transform(trainingSet);
        normalizer.transform(testSet);

        //System.out.println(testSet);

        ViewIterator trainIter = new ViewIterator(trainingSet, batchSize);
        ViewIterator testIter = new ViewIterator(testSet, batchSize);


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(lr))
                .weightInit(WeightInit.XAVIER)
                .l2(0.001)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(8)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(20)
                        .nOut(40)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(40)
                        .nOut(80)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nIn(80)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nIn(50)
                        .nOut(40)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new DenseLayer.Builder()
                        .nIn(40)
                        .nOut(30)
                        .activation(Activation.RELU)
                        .build())
                .layer(6, new DenseLayer.Builder()
                        .nIn(30)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(7, new DenseLayer.Builder()
                        .nIn(20)
                        .nOut(10)
                        .activation(Activation.RELU)
                        .build())
                .layer(8, new OutputLayer.Builder()
                        .nIn(10)
                        .nOut(1)
                        .activation(Activation.IDENTITY)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .build();

        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new StatsListener(storage, 1), new ScoreIterationListener(100));



        //  Fitting the model for nEpochs
        for(int i = 1; i <=  epoch; i++){
            System.out.println("Epoch: " + i);
            model.fit(trainIter, batchSize);
        }

        //  Evaluating the outcome of our trained model
        RegressionEvaluation regEval= model.evaluateRegression(testIter);
        System.out.println(regEval.stats());
    }

}
