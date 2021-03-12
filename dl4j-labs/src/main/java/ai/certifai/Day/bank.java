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
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class bank {

    static int seed = 123;
    static double lr = 0.01;
    static int epoch = 20;


    public static void main(String[] args) throws IOException, InterruptedException {
        File data = new ClassPathResource("bank/train.csv").getFile();
        FileSplit dataSplite = new FileSplit(data);
        CSVRecordReader csvdata = new CSVRecordReader(1, ',');
        csvdata.initialize(dataSplite);

        Schema sc =new Schema.Builder()
                .addColumnInteger("ID")
                .addColumnInteger("age")
                .addColumnCategorical("job", Arrays.asList("admin.", "unknown", "services", "management", "technician", "blue-collar", "retired", "housemaid", "self-employed", "student", "entrepreneur", "unemployed"))
                .addColumnCategorical("marital", Arrays.asList("married", "divorced", "single"))
                .addColumnCategorical("education", Arrays.asList("unknown", "secondary", "tertiary", "primary"))
                .addColumnCategorical("default", Arrays.asList("yes", "no"))
                .addColumnInteger("balance")
                .addColumnCategorical("housing", Arrays.asList("yes", "no"))
                .addColumnCategorical("loan", Arrays.asList("yes", "no"))
                .addColumnCategorical("contact", Arrays.asList("telephone", "cellular", "unknown"))
                .addColumnInteger("day")
                .addColumnCategorical("month", Arrays.asList("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"))
                .addColumnInteger("duration")
                .addColumnInteger("campaign")
                .addColumnInteger("pdays")
                .addColumnInteger("previous")
                .addColumnCategorical("poutcome", Arrays.asList("unknown", "success", "failure", "other"))
                .addColumnCategorical("subscribed", Arrays.asList("yes", "no"))
                .build();

        TransformProcess tp = new TransformProcess.Builder(sc)
                .removeColumns("ID")
                .removeColumns("day")
                .removeColumns("month")
                .categoricalToInteger("job")
                .categoricalToInteger("marital")
                .categoricalToInteger("education")
                .categoricalToInteger("default")
                .categoricalToInteger("housing")
                .categoricalToInteger("loan")
                .categoricalToInteger("contact")
                .categoricalToInteger("poutcome")
                .categoricalToInteger("subscribed")
                .build();

        System.out.println(tp.getFinalSchema());

        List<List<Writable>> alldata = new ArrayList<>();

        while(csvdata.hasNext()){
            alldata.add(csvdata.next());
        }

        List<List<Writable>> transformedData = LocalTransformExecutor.execute(alldata, tp);

//        for (List<Writable> transformedDatum : transformedData) {
//            System.out.println(transformedDatum);
//        }

        CollectionRecordReader collectionRR = new CollectionRecordReader(transformedData);

        DataSetIterator dataIter = new RecordReaderDataSetIterator(collectionRR, transformedData.size(),-1,2);

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

        NormalizerSerializer.getDefault().write(normalizer, "D:/TrainingLabs-main/dl4j-labs/src/main/java/ai/certifai/Day/bank.norm");

        System.out.println(testSet);

        ViewIterator trainIter = new ViewIterator(trainingSet, transformedData.size());
        ViewIterator testIter = new ViewIterator(testSet, transformedData.size());

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(lr))
                .weightInit(WeightInit.XAVIER)
                .l2(0.001)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(14)
                        .nOut(30)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(30)
                        .nOut(80)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(80)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nIn(50)
                        .nOut(40)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nIn(40)
                        .nOut(30)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new DenseLayer.Builder()
                        .nIn(30)
                        .nOut(10)
                        .activation(Activation.RELU)
                        .build())
                .layer(6, new OutputLayer.Builder()
                        .nIn(10)
                        .nOut(2)
                        .activation(Activation.SIGMOID)
                        .lossFunction(LossFunctions.LossFunction.XENT)
                        .build())
                .build();

        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new StatsListener(storage, 1));

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

        model.save(new File("D:/TrainingLabs-main/dl4j-labs/src/main/java/ai/certifai/Day/bank.model"));


    }
}
