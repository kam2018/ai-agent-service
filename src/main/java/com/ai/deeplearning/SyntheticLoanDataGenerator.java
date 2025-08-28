package com.ai.deeplearning;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SyntheticLoanDataGenerator {
    public static void main(String[] args) throws Exception {
        Random rand = new Random();
        List<String> rows = new ArrayList<>();
        rows.add("creditScore,income,loanAmount,employmentStatus,label");

        for (int i = 0; i < 1000; i++) {
            int creditScore = 300 + rand.nextInt(551); // 300–850
            int income = 10000 + rand.nextInt(190001); // ₹10k–₹200k
            int loanAmount = 5000 + rand.nextInt(95001); // ₹5k–₹100k
            int employmentStatus = rand.nextInt(3); // 0–2

            // Approval logic
            boolean approved = creditScore > 650 &&
                    income > 40000 &&
                    loanAmount < income * 0.5 &&
                    employmentStatus > 0;

            int label = approved ? 1 : 0;

            rows.add(String.format("%d,%d,%d,%d,%d", creditScore, income, loanAmount, employmentStatus, label));
            System.out.println(rows.get(i));
        }

    }

}
