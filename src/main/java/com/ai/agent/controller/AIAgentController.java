package com.ai.agent.controller;

import com.ai.agent.model.QueryRequest;
import com.ai.agent.model.QueryResponse;
import com.ai.agent.model.TrainRequest;
import com.ai.agent.service.AIAgentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/agent")
public class AIAgentController {
    private final AIAgentService aiAgentService;

    public AIAgentController(@Autowired AIAgentService aiAgentService) {
        this.aiAgentService = aiAgentService;
    }

    @PostMapping("/train")
    public ResponseEntity<String> train(@RequestBody TrainRequest request) {
        if (request.getDocumentText() == null || request.getDocumentText().isEmpty()) {
            return new ResponseEntity<>("Document text cannot be empty.", HttpStatus.BAD_REQUEST);
        }
        aiAgentService.trainAgent(request.getDocumentText(), request.getDocumentSource());
        return new ResponseEntity<>("Agent trained successfully!", HttpStatus.OK);
    }

    @PostMapping("/query")
    public ResponseEntity<QueryResponse> query(@RequestBody QueryRequest request) {
        if (request.getQuery() == null || request.getQuery().isEmpty()) {
            return new ResponseEntity<>(new QueryResponse("Query cannot be empty."), HttpStatus.BAD_REQUEST);
        }
        String response = aiAgentService.queryAgent(request.getQuery());
        return new ResponseEntity<>(new QueryResponse(response), HttpStatus.OK);
    }

    @PostMapping("/clear")
    public ResponseEntity<String> clear() {
        aiAgentService.clearTrainedData();
        return new ResponseEntity<>("Agent data cleared successfully!", HttpStatus.OK);
    }
}
