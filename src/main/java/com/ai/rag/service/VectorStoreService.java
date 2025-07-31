package com.ai.rag.service;

import org.springframework.ai.document.Document;
import org.springframework.ai.reader.TextReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.core.io.Resource;

import java.util.List;

public class VectorStoreService {
    private  final VectorStore vectorStore;

    private VectorStoreService(VectorStore vectorStore){
        this.vectorStore = vectorStore;
    }

    public static VectorStoreService of(VectorStore vectorStore){
        return new VectorStoreService(vectorStore);
    }

    public String load(Resource fileResource){
        TextReader textReader = new TextReader(fileResource);
        textReader.getCustomMetadata().put("filename", "Doc.txt");
        List<Document> documents = textReader.get();
        List<Document> documentsToken = new TokenTextSplitter().apply(documents);
        vectorStore.add(documentsToken);
        return "data initialised";

    }
}
