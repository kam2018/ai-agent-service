package com.ai.deeplearning;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.File;
import java.io.IOException;

@RestController
@RequestMapping("/loan")
public class MyModelController {

    private final MultiLayerNetwork model;
    private final NormalizerStandardize normalizer;

    public MyModelController() throws Exception {
        // Load trained model and normalizer
        model = MultiLayerNetwork.load(new File("loan_approval_model.zip"), true);
        normalizer = NormalizerSerializer.getDefault().restore(new File("normalizer.bin"));
    }

    @PostMapping("/approve")
    public String approveLoan(@RequestBody LoanApplication loanApplication) throws JsonProcessingException {
        // Prepare input data
        INDArray input = Nd4j.create(new double[]{
                loanApplication.getCreditScore(),
                loanApplication.getIncome(),
                loanApplication.getLoanAmount(),
                loanApplication.getEmploymentStatus()
        }, 1, 4);

        // Make prediction
        normalizer.transform(input); // Normalize input
        INDArray output = model.output(input);
        ObjectMapper objectMapper = new ObjectMapper();
        String jsonString = objectMapper.writeValueAsString(output);

        System.out.println("\nSerialized JSON String:");
        System.out.println(jsonString);
        //int prediction = Nd4j.argMax(output, 1).getInt(0);
        System.out.println("Normalized input: " + input);
        System.out.println("Model output: " + output);
        if (output.getDouble(1) > 0.45) {
            return "Approved";
        } else {
            return "Rejected";
        }


        //return prediction == 1 ? "Approved ✅" : "Rejected ❌";
    }
}
