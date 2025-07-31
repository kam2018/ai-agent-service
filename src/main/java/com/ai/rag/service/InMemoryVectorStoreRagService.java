package com.ai.rag.service;

import com.ai.rag.utils.QuestionAnswerAdvisorUtils;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;

@Service
public class InMemoryVectorStoreRagService {
    @Value("classpath:/Doc.txt")
    private Resource fileResource;
    private final ChatClient chatClient;
    private final VectorStore simpleVectorStore;

    public InMemoryVectorStoreRagService(ChatClient.Builder chatClientBuilder, VectorStore simpleVectorStore){
        this.chatClient = chatClientBuilder.build();
        this.simpleVectorStore = simpleVectorStore;
    }

    public ChatResponse getQueryResponse(String query){
        VectorStoreService.of(simpleVectorStore).load(fileResource);
        return this.chatClient.prompt()
                .advisors(QuestionAnswerAdvisorUtils.getSimpleQuestionAnsAdvisor(simpleVectorStore))
                .user(query)
                .call()
                .chatResponse();
    }
}
