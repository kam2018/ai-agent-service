package com.ai.deeplearning.response;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.categorical.CategoricalToIntegerTransform;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.transform.transform.categorical.StringToCategoricalTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.evaluation.classification.Evaluation;

import java.io.File;
import java.util.Arrays;

public class MyModelTextResponse {
    public static void main(String[] args) throws Exception {
        int batchSize = 8;
        int labelIndex = 4;
        int numClasses = 5; // Updated to match normalized label categories

        // Define schema
        Schema inputSchema = new Schema.Builder()
                .addColumnInteger("creditScore")
                .addColumnInteger("income")
                .addColumnInteger("loanAmount")
                .addColumnInteger("employmentStatus")
                .addColumnString("label")
                .build();

        // Define transform process with normalized labels
        TransformProcess tp = new TransformProcess.Builder(inputSchema)
                .transform(new StringToCategoricalTransform("label", Arrays.asList(
                        "Approved",
                        "Rejected: credit score < 400",
                        "Rejected: income < loanAmount",
                        "Rejected: loan > 50% of income",
                        "Rejected: employment status ≠ 2"
                )))

                .transform(new CategoricalToIntegerTransform("label"))
                .build();

        // Read and transform CSV
        CSVRecordReader csvReader = new CSVRecordReader(1, ",");
        csvReader.initialize(new FileSplit(new File("synthetic_loan_text_response_data.csv")));
        TransformProcessRecordReader tpReader = new TransformProcessRecordReader(csvReader, tp);

        DataSetIterator iterator = new RecordReaderDataSetIterator(tpReader, batchSize, labelIndex, numClasses);
        DataSet allData = iterator.next();
        allData.shuffle();

        // Normalize
        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(allData);
        normalizer.transform(allData);

        // Split
        SplitTestAndTrain split = allData.splitTestAndTrain(0.75);
        DataSet trainData = split.getTrain();
        DataSet testData = split.getTest();

        // Network config
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.01))
                .list()
                .layer(new DenseLayer.Builder().nIn(4).nOut(10).activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(10).nOut(5).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX).nIn(5).nOut(numClasses).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        // Train
        for (int i = 0; i < 100; i++) {
            model.fit(trainData);
        }

        // Evaluate
        Evaluation eval = new Evaluation(numClasses);
        eval.eval(testData.getLabels(), model.output(testData.getFeatures()));
        System.out.println(eval.stats());

        // Save model
        model.save(new File("loan_approval_model_text_response.zip"), true);

        // Map predictions back to text
        int[] predictedClasses = model.predict(testData.getFeatures());
        String[] labelMap = {
                "Approved",
                "Rejected: credit score < 400",
                "Rejected: income < loanAmount",
                "Rejected: loan > 50% of income",
                "Rejected: employment status ≠ 2"
        };


        System.out.println("\nPredictions:");
        for (int i = 0; i < predictedClasses.length; i++) {
            System.out.println("Sample " + i + ": " + labelMap[predictedClasses[i]]);
        }

        int[] predictedClasses1 = model.predict(testData.getFeatures());
        INDArray actualLabels = testData.getLabels();

        for (int i = 0; i < predictedClasses1.length; i++) {
            int actual = Nd4j.argMax(actualLabels.getRow(i), 1).getInt(0);
            int predicted = predictedClasses1[i];
            System.out.println("Sample " + i + ": Actual = " + labelMap[actual] + ", Predicted = " + labelMap[predicted]);
        }
    }
}