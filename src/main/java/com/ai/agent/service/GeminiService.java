package com.ai.agent.service;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class GeminiService {
    private final ChatClient chatClient;

    public GeminiService(ChatClient chatClient) {
        this.chatClient = chatClient;
    }

    public <T> List<T> getResponse(PromptTemplate template, ParameterizedTypeReference<List<T>> typeReference){
        return this.chatClient.prompt(template.create())
                .call()
                .entity(typeReference);
    }
}
