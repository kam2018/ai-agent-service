package com.ai.rag.service;

import org.apache.el.stream.Stream;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.reader.TextReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.core.io.Resource;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class VectorStoreService {
    private  final VectorStore vectorStore;

    private EmbeddingModel embeddingModel;

    private VectorStoreService(VectorStore vectorStore){
        this.vectorStore = vectorStore;
    }
    private VectorStoreService(VectorStore vectorStore, EmbeddingModel embeddingModel){
        this.vectorStore = vectorStore;
        this.embeddingModel = embeddingModel;
    }

    public static VectorStoreService of(VectorStore vectorStore){
        return new VectorStoreService(vectorStore);
    }

    public static VectorStoreService of(VectorStore vectorStore, EmbeddingModel embeddingModel){
        return new VectorStoreService(vectorStore, embeddingModel);
    }

    public String load(Resource fileResource){
        TextReader textReader = new TextReader(fileResource);
        textReader.getCustomMetadata().put("filename", "Doc.txt");
        List<Document> documents = textReader.get();
        List<Document> documentsToken = new TokenTextSplitter().apply(documents);
        vectorStore.add(documentsToken);
        return "data initialised with keyWord:" + documentsToken.size();
    }

    public String semanticSearch(String query) {
        return vectorStore.similaritySearch(SearchRequest.builder()
                        .query(query)
                        .similarityThreshold(0.8)
                        .topK(1)
                        .build())
                .stream()
                .map(Document::getText)
                .reduce("", (a, b) -> a + b + "\n");
    }
}
