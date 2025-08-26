package com.ai.rag.api;

import com.ai.rag.service.InMemoryVectorStoreRagService;
import com.ai.rag.service.VectorStoreService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/vector/db")
@Slf4j
public class VectorDbController {

    @Value("classpath:/Doc.txt")
    private Resource fileResource;
    private final VectorStoreService vectorStoreService;

    public VectorDbController(VectorStore simpleVectorStore, EmbeddingModel embeddingModel){
        this.vectorStoreService = VectorStoreService.of(simpleVectorStore, embeddingModel);
    }

    @GetMapping("/load")
    public String load(){
        log.info("Resource loading started !!!!");
        return vectorStoreService.load(fileResource);
    }

    @GetMapping("/query")
    public String queryFromVector(@RequestParam String query) {
        String search = vectorStoreService.semanticSearch(query);
        log.info("generate response {}", search);
        return search;
    }
}
