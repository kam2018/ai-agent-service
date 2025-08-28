package com.ai.deeplearning;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.evaluation.classification.Evaluation;

import java.io.File;

public class MyModel {
    public static void main(String[] args) throws Exception {
        // Load dataset
        int labelIndex = 4; // Index of the label column
        int numClasses = 2; // Approve or Reject
        int batchSize = 64;

        CSVRecordReader recordReader = new CSVRecordReader(1, ",");
        recordReader.initialize(new FileSplit(new File("synthetic_loan_data.csv")));
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);

        // Load all data into memory
        DataSet allData = iterator.next();
        allData.shuffle();

        // Normalize the data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(allData);
        normalizer.transform(allData);

        // Split into training and test sets
        SplitTestAndTrain split = allData.splitTestAndTrain(0.8);
        DataSet trainData = split.getTrain();
        DataSet testData = split.getTest();

        // Define the network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.01))
                .list()
                .layer(new DenseLayer.Builder().nIn(4).nOut(10)
                        .activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(10).nOut(5)
                        .activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(5).nOut(numClasses).build())
                .backpropType(BackpropType.Standard)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        // Train the model
        for (int i = 0; i < 100; i++) {
            model.fit(trainData);
            NormalizerSerializer.getDefault().write(normalizer, new File("normalizer.bin"));
        }

        // Evaluate the model
        Evaluation eval = new Evaluation(numClasses);
        eval.eval(testData.getLabels(), model.output(testData.getFeatures()));
        System.out.println(eval.stats());

        // Save the model
        model.save(new File("loan_approval_model.zip"), true);
    }
}