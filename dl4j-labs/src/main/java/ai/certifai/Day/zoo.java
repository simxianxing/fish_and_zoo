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
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.DataSet;
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

public class zoo {

    private static int seed = 123;
    private static int epoch = 10;
    private static double lr = 0.01;
    private static int batchsize = 10;

    public static void main(String[] args) throws IOException, InterruptedException {

        File data = new ClassPathResource("zoo/zoo.csv").getFile();
        FileSplit dataSplite = new FileSplit(data);
        CSVRecordReader csvdata = new CSVRecordReader(1, ',');
        csvdata.initialize(dataSplite);

        Schema sc =new Schema.Builder()
                .addColumnCategorical("Species", Arrays.asList("aardvark", "antelope", "bear", "boar", "buffalo", "calf", "cavy", "cheetah", "deer", "dolphin", "elephant", "fruitbat", "giraffe", "girl", "goat", "gorilla", "hamster", "hare", "leopard", "lion", "lynx", "mink", "mole", "mongoose", "opossum", "oryx", "platypus", "polecat", "pony", "porpoise", "puma", "pussycat", "raccoon", "reindeer", "seal", "sealion", "squirrel", "vampire", "vole", "wallaby", "wolf", "chicken", "crow", "dove", "duck", "flamingo", "gull", "hawk", "kiwi", "lark", "ostrich", "parakeet", "penguin", "pheasant", "rhea", "skimmer", "skua", "sparrow", "swan", "vulture", "wren", "pitviper", "seasnake", "slowworm", "tortoise", "tuatara", "bass", "carp", "catfish", "chub", "dogfish", "haddock", "herring", "pike", "piranha", "seahorse", "sole", "stingray", "tuna", "frog", "frog", "newt", "toad", "flea", "gnat", "honeybee", "housefly", "ladybird", "moth", "termite", "wasp", "clam", "crab", "crayfish", "lobster", "octopus", "scorpion", "seawasp", "slug", "starfish", "worm"))
                .addColumnInteger("hair")
                .addColumnInteger("feathers")
                .addColumnInteger("eggs")
                .addColumnInteger("milk")
                .addColumnInteger("airborne")
                .addColumnInteger("aquatic")
                .addColumnInteger("predator")
                .addColumnInteger("toothed")
                .addColumnInteger("backbone")
                .addColumnInteger("breathes")
                .addColumnInteger("venomous")
                .addColumnInteger("fins")
                .addColumnInteger("legs")
                .addColumnInteger("tail")
                .addColumnInteger("domestic")
                .addColumnInteger("catsize")
                .addColumnCategorical("class_type", Arrays.asList("1", "2", "3", "4", "5", "6", "7"))
                .build();

        TransformProcess tp = new TransformProcess.Builder(sc)
                .removeColumns("Species")
                .categoricalToInteger("class_type")
                .build();

        System.out.println(tp.getFinalSchema());

        List<List<Writable>> alldata = new ArrayList<>();

        while(csvdata.hasNext()){
            alldata.add(csvdata.next());
        }

        //System.out.println(alldata);

        List<List<Writable>> transformedData = LocalTransformExecutor.execute(alldata, tp);

        CollectionRecordReader collectionRR = new CollectionRecordReader(transformedData);

        DataSetIterator dataIter = new RecordReaderDataSetIterator(collectionRR, transformedData.size(), -1, 7);

        DataSet allData = dataIter.next();
        allData.shuffle();
        System.out.println(allData);
//
        SplitTestAndTrain testTrainSplit = allData.splitTestAndTrain(0.75);

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

        NormalizerSerializer.getDefault().write(normalizer, "D:/TrainingLabs-main/dl4j-labs/src/main/java/ai/certifai/Day/zoo.norm");

        System.out.println(testSet);

        ViewIterator trainIter = new ViewIterator(trainingSet, batchsize);
        ViewIterator testIter = new ViewIterator(testSet, batchsize);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(lr))
                .weightInit(WeightInit.XAVIER)
                .l2(0.001)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(16)
                        .nOut(30)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(30)
                        .nOut(40)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(40)
                        .nOut(30)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nIn(30)
                        .nOut(10)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nIn(30)
                        .nOut(10)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new OutputLayer.Builder()
                        .nIn(10)
                        .nOut(7)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .build();

        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new StatsListener(storage, 1), new ScoreIterationListener(10));

        Evaluation eval;
        for(int i = 1; i <=  epoch; i++){
            model.fit(trainIter, batchsize);
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


        model.save(new File("D:/TrainingLabs-main/dl4j-labs/src/main/java/ai/certifai/Day/zoo_2.model"));




    }
}
