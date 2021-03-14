package ai.certifai.Day;


import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class loadbank {

    public static void main(String[] args) throws Exception {

        File data = new ClassPathResource("bank/test.csv").getFile();
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

        System.out.println("collectionRR.getLabels() = " + collectionRR.getLabels());

        DataSetIterator dataIter = new RecordReaderDataSetIterator(collectionRR, transformedData.size());

        DataSet allData = dataIter.next();

        DataNormalization  normalizer = NormalizerSerializer.getDefault().restore("D:/TrainingLabs-main/dl4j-labs/src/main/java/ai/certifai/Day/bank.norm");
        normalizer.transform(allData);

        System.out.println(allData);

        ViewIterator trainIter = new ViewIterator(allData, transformedData.size());

        MultiLayerNetwork model = MultiLayerNetwork.load(new File("D:/TrainingLabs-main/dl4j-labs/src/main/java/ai/certifai/Day/bank.model"), true);

        INDArray prediction = model.output(trainIter);
        INDArray pre = Nd4j.argMax(prediction, 1);
        System.out.println(pre);

        DataBuffer dataBuffer = pre.data();
        int[] array = dataBuffer.asInt();

        System.out.println(array.length);


        File file = new File("D:/TrainingLabs-main/dl4j-labs/src/main/java/ai/certifai/Day/bank.txt");
        file.createNewFile();

        FileWriter outputfile = new FileWriter("D:/TrainingLabs-main/dl4j-labs/src/main/java/ai/certifai/Day/bank.txt");

        for (int j = 0; j < array.length; j++) {
            outputfile.write(String.valueOf(array[j]));
            outputfile.write("\n");
        }
        outputfile.close();


    }
}
