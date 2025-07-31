package com.ai.rag.api;

import com.ai.rag.service.PgVectorRagService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/rag")
@Slf4j
public class PgVectorRagController {
    private final PgVectorRagService pgVectorRagService;

    public PgVectorRagController(PgVectorRagService pgVectorRagService) {
        this.pgVectorRagService = pgVectorRagService;
    }

    @GetMapping("/query/db")
    public ChatResponse queryDb(@RequestParam String query) {
        ChatResponse queryResponse = pgVectorRagService.getQueryResponse(query);

        log.info("generate response {}", queryResponse);
        return queryResponse;
    }
}
