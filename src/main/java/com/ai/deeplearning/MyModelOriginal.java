package com.ai.deeplearning;

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
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;

import java.io.File;

public class MyModelOriginal {
    public static void main(String[] args) throws Exception {
        // Load dataset
        CSVRecordReader recordReader = new CSVRecordReader(1, ",");
        recordReader.initialize(new FileSplit(new ClassPathResource("loan_data.csv").getFile()));

        int labelIndex = 4; // Index of the label (approve/reject)
        int numClasses = 2; // Approve or Reject
        int batchSize = 64;
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);

        // Normalize the data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(iterator);
        iterator.setPreProcessor(normalizer);

        // Define the network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .maxNumLineSearchIterations(1000)
                //.iterations(1000)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.01))
                //.learningRate(0.01)
                .list()

                .layer(new DenseLayer.Builder().nIn(4).nOut(3).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(2).build())
                .backpropType(BackpropType.Standard)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        // Train the model
        for (int i = 0; i < 1000; i++) {
            iterator.reset();
            model.fit(iterator);
        }

        // Save the model
        model.save(new File("loan_approval_model.zip"), true);
    }
}
