package com.ai.agent.service;

import com.ai.agent.config.AppConfig;
import com.ai.agent.model.LLMRequest;
import com.ai.agent.model.LLMResponse;
import com.ai.agent.model.Part;
import com.ai.agent.model.RequestContent;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.web.client.RestTemplate;

import java.util.Collections;

public class LLMService {
    private final AppConfig appConfig;
    private final RestTemplate restTemplate;

    public LLMService(@Autowired AppConfig appConfig, @Autowired RestTemplate restTemplate) {
        this.appConfig = appConfig;
        this.restTemplate = restTemplate;
    }

    public String generateResponse(String prompt) {
        // Construct the request body for the Gemini API
        Part part = new Part(prompt);
        RequestContent content = new RequestContent("user", Collections.singletonList(part));
        LLMRequest request = new LLMRequest(Collections.singletonList(content));

        // Set HTTP headers
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        // Create the HTTP entity (request body + headers)
        HttpEntity<LLMRequest> entity = new HttpEntity<>(request, headers);

        try {
            // Make the POST request to the Gemini LLM API
            // The API URL includes the model and the API key as a query parameter
            String url = appConfig.getEmbeddingApiUrl() + "?key=" + appConfig.getApiKey();
            LLMResponse response = restTemplate.postForObject(url, entity, LLMResponse.class);

            // Extract the generated text from the response
            if (response != null && response.getCandidates() != null && !response.getCandidates().isEmpty() &&
                    response.getCandidates().get(0).getContent() != null &&
                    response.getCandidates().get(0).getContent().getParts() != null &&
                    !response.getCandidates().get(0).getContent().getParts().isEmpty()) {
                return response.getCandidates().get(0).getContent().getParts().get(0).getText();
            }
        } catch (Exception e) {
            System.err.println("Error calling Gemini LLM API: " + e.getMessage());
            e.printStackTrace();
        }
        return null;
    }
}
