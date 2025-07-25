package com.ai.agent.service;

import com.ai.agent.model.DocumentChunk;
import com.ai.agent.storage.VectorStore;

import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

public class AIAgentService {
    private final DocumentProcessor documentProcessor;
    private final EmbeddingService embeddingService;
    private final VectorStore vectorStore;
    private final LLMService llmService;

    // Number of top relevant chunks to retrieve for RAG
    private static final int TOP_K_CHUNKS = 3;

    public AIAgentService(DocumentProcessor documentProcessor,
                          EmbeddingService embeddingService,
                          VectorStore vectorStore,
                          LLMService llmService) {
        this.documentProcessor = documentProcessor;
        this.embeddingService = embeddingService;
        this.vectorStore = vectorStore;
        this.llmService = llmService;
    }

    public void trainAgent(String documentText, String documentSource) {
        System.out.println("Starting training for document: " + documentSource);

        List<String> textChunks = documentProcessor.splitDocument(documentText);
        System.out.println("Document split into " + textChunks.size() + " chunks.");

        List<DocumentChunk> documentChunks = textChunks.stream()
                .map(chunkText -> {
                    // 2. Generate embedding for each chunk
                    List<Float> embedding = embeddingService.getEmbedding(chunkText);
                    if (embedding == null) {
                        System.err.println("Failed to generate embedding for a chunk. Skipping.");
                        return null; // Skip chunks that failed embedding
                    }
                    // Create a DocumentChunk object with a unique ID
                    return new DocumentChunk(UUID.randomUUID().toString(), chunkText, embedding, documentSource);
                })
                .filter(java.util.Objects::nonNull) // Filter out nulls from failed embeddings
                .collect(Collectors.toList());

        // 3. Add the processed chunks to the vector store
        if (!documentChunks.isEmpty()) {
            vectorStore.addDocumentChunks(documentChunks);
            System.out.println("Successfully trained agent with " + documentChunks.size() + " document chunks from " + documentSource);
        } else {
            System.out.println("No valid chunks generated or embedded for training from " + documentSource);
        }
    }

    public String queryAgent(String query) {
        System.out.println("Processing query: " + query);

        // 1. Generate embedding for the query
        List<Float> queryEmbedding = embeddingService.getEmbedding(query);
        if (queryEmbedding == null) {
            return "Error: Could not generate embedding for the query.";
        }

        // 2. Retrieve top K relevant document chunks from the vector store
        List<DocumentChunk> relevantChunks = vectorStore.similaritySearch(queryEmbedding, TOP_K_CHUNKS);
        System.out.println("Found " + relevantChunks.size() + " relevant chunks.");

        // 3. Construct a prompt for the LLM using the query and retrieved context
        String context = relevantChunks.stream()
                .map(DocumentChunk::getText)
                .collect(Collectors.joining("\n---\n")); // Join chunks with a separator

        String llmPrompt;
        if (!context.isEmpty()) {
            llmPrompt = String.format(
                    "Based on the following context, answer the question. If the answer is not in the context, state that you don't know.\n\nContext:\n%s\n\nQuestion: %s",
                    context, query
            );
        } else {
            llmPrompt = String.format(
                    "Answer the following question. If you don't know the answer, state that you don't know.\n\nQuestion: %s",
                    query
            );
            System.out.println("No relevant context found for the query. Answering based on general knowledge.");
        }

        // 4. Generate response using the LLM
        String llmResponse = llmService.generateResponse(llmPrompt);

        if (llmResponse == null) {
            return "Error: Could not generate a response from the AI model.";
        }

        System.out.println("LLM Response generated.");
        return llmResponse;
    }

    public void clearTrainedData() {
        vectorStore.clear();
        System.out.println("Agent's trained data cleared.");
    }

}
