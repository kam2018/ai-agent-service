package com.ai.rag.service;

import com.ai.rag.utils.QuestionAnswerAdvisorUtils;
import jakarta.annotation.PostConstruct;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;

@Service
public class PgVectorRagService {
    @Value("classpath:/Doc.txt")
    private Resource fileResource;
    private final ChatClient chatClient;

    private final PgVectorStore vectorStore;


    public PgVectorRagService(ChatClient.Builder chatClientBuilder, PgVectorStore vectorStore){
        this.chatClient = chatClientBuilder.build();
        this.vectorStore = vectorStore;
    }

    @PostConstruct
    public void init(){
        VectorStoreService.of(vectorStore).load(fileResource);
    }

    public ChatResponse getQueryResponse(String query){
        return this.chatClient.prompt()
                .advisors(QuestionAnswerAdvisorUtils.getSimpleQuestionAnsAdvisor(vectorStore))
                .user(query)
                .call()
                .chatResponse();
    }

}
