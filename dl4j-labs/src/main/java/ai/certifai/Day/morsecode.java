package ai.certifai.Day;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.awt.event.KeyEvent;
import java.io.File;
import java.util.Arrays;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.helper.opencv_core.RGB;

public class morsecode {
    //private static final Logger log = org.slf4j.LoggerFactory.getLogger(EditAtBottleneckAndExtendModel.class);

    private static double trainPerc = 0.7;
    private static int width = 416;
    private static int height = 416;
    private static int channels = 3;
    private static int batchSize = 50;
    private static int numClass = 5;
    private static int numBox = 5;
    private static int epoch = 1;
    private static int seed = 123;
    private static double lr = 0.01;
    private static double detectionThreshold = 0.5;
    private static final Scalar GREEN = RGB(0, 255.0, 0);
    private static final Scalar YELLOW = RGB(255, 255, 0);
    private static Scalar[] colormap = {GREEN, YELLOW};
    private static String labeltext = null;
    private static Frame frame = null;

    private static ComputationGraph model;
    private static List<String> labels;


    private static double[][] priorBoxes = {{1, 3}, {2.5, 6}, {3, 4}, {3.5, 8}, {4, 9}};


    private static String modelfilename = "D:\\TrainingLabs-main\\dl4j-labs\\src\\main\\java\\ai\\certifai\\Day\\morsecode.zip";

    public static void main(String[] args) throws Exception {
        File myFile = new ClassPathResource("morseCodeDecoder").getFile();
        System.out.println(myFile);
        //Image Augmentation
        ImageTransform hFlip = new FlipImageTransform(1);

        //Image transform method, probability of images to get transform
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(new Pair<>(hFlip, 0.2));
        PipelineImageTransform tp = new PipelineImageTransform(pipeline, false);

        moesecodeiterator iterator = new moesecodeiterator();
        iterator.setup(myFile, trainPerc, width, height, channels, batchSize, numClass, tp);

        RecordReaderDataSetIterator trainIter = iterator.getTrain(batchSize);
        RecordReaderDataSetIterator testIter = iterator.getTest(batchSize);

        labels = trainIter.getLabels();
        System.out.println(labels);

        if (new File(modelfilename).exists()) {
            //        STEP 2 : Load trained model from previous execution
            Nd4j.getRandom().setSeed(seed);
            System.out.println("Load model...");
            model = ModelSerializer.restoreComputationGraph(modelfilename);
        } else{
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

            model = new TransferLearning.GraphBuilder(pretrained)
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

            StatsStorage storage = new InMemoryStatsStorage();
            UIServer server = UIServer.getInstance();
            server.attach(storage);
            model.setListeners(new StatsListener(storage, 1), new ScoreIterationListener(1));

            for(int i = 1; i <=  epoch; i++){
                trainIter.reset();
                while (trainIter.hasNext()){
                    model.fit(trainIter.next());
                }
                System.out.println("\nCompleted epoch " + i + "\n");
            }
            ModelSerializer.writeModel(model, modelfilename, true);
            System.out.println("Model saved.");
        }


        //Evaluate the model's accuracy by using the test iterator.
        //OfflineValidationWithTestDataset(testIter);
        //Inference the model and process the webcam stream and make predictions.
        doInference();

    }

//    private static void OfflineValidationWithTestDataset(RecordReaderDataSetIterator test) throws InterruptedException {
//        NativeImageLoader imageLoader = new NativeImageLoader();
//        CanvasFrame canvas = new CanvasFrame("Validate Test Dataset");
//        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
//        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
//        Mat convertedMat = new Mat();
//        Mat convertedMat_big = new Mat();
//
//        while (test.hasNext() && canvas.isVisible()) {
//            org.nd4j.linalg.dataset.DataSet ds = test.next();
//            INDArray features = ds.getFeatures();
//            INDArray results = model.outputSingle(features);
//            List<DetectedObject> objs = yout.getPredictedObjects(results, detectionThreshold);
//            YoloUtils.nms(objs, 0.4);
//            Mat mat = imageLoader.asMat(features);
//            mat.convertTo(convertedMat, CV_8U, 255, 0);
//            int w = mat.cols() * 2;
//            int h = mat.rows() * 2;
//            resize(convertedMat, convertedMat_big, new Size(w, h));
//            convertedMat_big = drawResults(objs, convertedMat_big, w, h);
//            canvas.showImage(converter.convert(convertedMat_big));
//            canvas.waitKey();
//        }
//        canvas.dispose();
//    }

    private static Mat drawResults(List<DetectedObject> objects, Mat mat, int w, int h) {
        for (DetectedObject obj : objects) {
            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();
            String label = labels.get(obj.getPredictedClass());
            int x1 = (int) Math.round(w * xy1[0] / moesecodeiterator.gridWidth);
            int y1 = (int) Math.round(h * xy1[1] / moesecodeiterator.gridHeight);
            int x2 = (int) Math.round(w * xy2[0] / moesecodeiterator.gridWidth);
            int y2 = (int) Math.round(h * xy2[1] / moesecodeiterator.gridHeight);
            //Draw bounding box
            rectangle(mat, new Point(x1, y1), new Point(x2, y2), colormap[obj.getPredictedClass()], 2, 0, 0);
            //Display label text
            labeltext = label + " " + String.format("%.2f", obj.getConfidence() * 100) + "%";
            int[] baseline = {0};
            Size textSize = getTextSize(labeltext, FONT_HERSHEY_DUPLEX, 1, 1, baseline);
            rectangle(mat, new Point(x1 + 2, y2 - 2), new Point(x1 + 2 + textSize.get(0), y2 - 2 - textSize.get(1)), colormap[obj.getPredictedClass()], FILLED, 0, 0);
            putText(mat, labeltext, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, RGB(0, 0, 0));


        }
        return mat;
    }

    private static void doInference() {

        String cameraPos = "front";
        int cameraNum = 0;
        Thread thread = null;
        NativeImageLoader loader = new NativeImageLoader(
                width,
                height,
                channels,
                new ColorConversionTransform(COLOR_BGR2RGB));
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);

        if (!cameraPos.equals("front") && !cameraPos.equals("back")) {
            try {
                throw new Exception("Unknown argument for camera position. Choose between front and back");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        FrameGrabber grabber = null;
        try {
            grabber = FrameGrabber.createDefault(cameraNum);
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        try {
            grabber.start();
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }

        CanvasFrame canvas = new CanvasFrame("Object Detection");
        int w = grabber.getImageWidth();
        int h = grabber.getImageHeight();
        canvas.setCanvasSize(w, h);

        while (true) {
            try {
                frame = grabber.grab();
            } catch (FrameGrabber.Exception e) {
                e.printStackTrace();
            }

            //if a thread is null, create new thread
            if (thread == null) {
                thread = new Thread(() ->
                {
                    while (frame != null) {
                        try {
                            Mat rawImage = new Mat();

                            //Flip the camera if opening front camera
                            if (cameraPos.equals("front")) {
                                Mat inputImage = converter.convert(frame);
                                flip(inputImage, rawImage, 1);
                            } else {
                                rawImage = converter.convert(frame);
                            }

                            Mat resizeImage = new Mat();
                            resize(rawImage, resizeImage, new Size(width, height));
                            INDArray inputImage = loader.asMatrix(resizeImage);
                            scaler.transform(inputImage);
                            INDArray outputs = model.outputSingle(inputImage);
                            org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
                            List<DetectedObject> objs = yout.getPredictedObjects(outputs, detectionThreshold);
                            YoloUtils.nms(objs, 0.4);
                            rawImage = drawResults(objs, rawImage, w, h);
                            canvas.showImage(converter.convert(rawImage));

                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                });
                thread.start();
            }

            KeyEvent t = null;
            try {
                t = canvas.waitKey(33);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            if ((t != null) && (t.getKeyCode() == KeyEvent.VK_Q)) {
                break;
            }
        }
    }


}
