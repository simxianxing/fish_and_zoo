package ai.certifai.Day;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class newnetwork {

    private static Logger log = LoggerFactory.getLogger(newnetwork.class);
    public static void main(String[] args) throws InterruptedException {
        //create two arrays
        //input
        INDArray inp = Nd4j.create(new float[]{(float) 0.2}, 1,1);
        System.out.println(inp);

        //output
        INDArray out = Nd4j.create(new float[]{(float) 0.7}, 1,1);
        System.out.println(out);

        //configure the neural network
        //input = 1
        //output = 1
        //hidden layer = 2
        //hidden neuron = 5
        //loss function = MSE / RMSE / MAE / R2
        //activation function hidden layer = sigmoid
        //activation function output layer = identity
        //learning rate = 0.001 / 1
        //weight loss = XAVIER
        //dropout = 0.5
        //regularization = L1 / L2 = 0.2
        //seed = 123

        int numofinput = 1;
        int numofoutput = 1;
        int numHidden1 = 2;
        int numHidden2 = 2;
        int epochs = 100;
        int seed = 123;
        double lr = 0.01;

        MultiLayerConfiguration mylayer = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(lr))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numofinput)
                        .nOut(numHidden1)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(numHidden1)
                        .nOut(numHidden2)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(2, new OutputLayer.Builder()
                        .nIn(numHidden2)
                        .nOut(numofoutput)
                        .activation(Activation.IDENTITY)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .build();

        //UI server
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        //initialize the network
        MultiLayerNetwork mynetwork = new MultiLayerNetwork(mylayer);
        mynetwork.init();
        mynetwork.setListeners(new StatsListener(storage, 1));

        //initialize the model
        for(int i = 0; i < epochs; i++){
            log.info("Epoch " + i);
            mynetwork.fit(inp, out);

            INDArray predicted = mynetwork.output(inp);
            log.info("predicted: " + predicted.toString());

            //Thread.sleep(100);
        }

        //evaluate the model
    }

}
