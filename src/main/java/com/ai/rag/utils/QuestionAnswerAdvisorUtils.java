package com.ai.rag.utils;

import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;

public class QuestionAnswerAdvisorUtils {

    public static QuestionAnswerAdvisor getSimpleQuestionAnsAdvisor(VectorStore vectorStore){
        return QuestionAnswerAdvisor.builder(vectorStore)
                .searchRequest(SearchRequest.builder().build())
                .build();
    }
}
