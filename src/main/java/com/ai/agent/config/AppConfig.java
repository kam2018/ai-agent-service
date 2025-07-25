package com.ai.agent.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

@Configuration
public class AppConfig {
    @Value("${gemini.api.key}")
    private String apiKey;
    @Value("${gemini.embedding.url}")
    private String embeddingApiUrl;

    public String getApiKey() {
        return apiKey;
    }

    public String getEmbeddingApiUrl() {
        return embeddingApiUrl;
    }

    @Bean
    public RestTemplate restTemplate(){
        return new RestTemplate();
    }


}
