package com.ai.agent.service;

import com.ai.agent.config.AppConfig;
import com.ai.agent.model.EmbeddingRequest;
import com.ai.agent.model.EmbeddingResponse;
import com.ai.agent.model.Part;
import com.ai.agent.model.RequestContent;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.Collections;
import java.util.List;

@Service
public class EmbeddingService {
    private final AppConfig appConfig;
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;

    public EmbeddingService(@Autowired AppConfig appConfig, @Autowired RestTemplate restTemplate, @Autowired ObjectMapper objectMapper) {
        this.appConfig = appConfig;
        this.restTemplate = restTemplate;
        this.objectMapper = objectMapper;
    }

    public List<Float> getEmbedding(String text) {
        // Construct the request body for the Gemini API
        Part part = new Part(text);
        RequestContent content = new RequestContent("user", Collections.singletonList(part));
        EmbeddingRequest request = new EmbeddingRequest(Collections.singletonList(content));

        // Set HTTP headers
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        // Create the HTTP entity (request body + headers)
        HttpEntity<EmbeddingRequest> entity = new HttpEntity<>(request, headers);

        try {
            // Make the POST request to the Gemini embedding API
            // The API URL includes the model and the API key as a query parameter
            String url = appConfig.getEmbeddingApiUrl() + "?key=" + appConfig.getApiKey();
            EmbeddingResponse response = restTemplate.postForObject(url, entity, EmbeddingResponse.class);

            // Extract the embedding from the response.
            // The Gemini API returns the embedding as a JSON string within the 'text' field.
            if (response != null && response.getCandidates() != null && !response.getCandidates().isEmpty() &&
                    response.getCandidates().get(0).getContent() != null &&
                    response.getCandidates().get(0).getContent().getParts() != null &&
                    !response.getCandidates().get(0).getContent().getParts().isEmpty()) {

                String embeddingJson = response.getCandidates().get(0).getContent().getParts().get(0).getText();
                // Parse the JSON string into a List<Float>
                return objectMapper.readValue(embeddingJson, new TypeReference<List<Float>>() {
                });
            }
        } catch (JsonProcessingException e) {
            System.err.println("Error parsing embedding JSON: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Error calling Gemini Embedding API: " + e.getMessage());
            e.printStackTrace();
        }
        return null;
    }
}
