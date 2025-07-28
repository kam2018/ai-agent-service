package com.ai.agent.storage;

import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

@Component
public class InMemoryVectorStore implements VectorStore {
    private final ConcurrentMap<String, DocumentChunk> chunks = new ConcurrentHashMap<>();

    @Override
    public void addDocumentChunks(List<DocumentChunk> chunksToAdd) {
        chunksToAdd.forEach(chunk -> chunks.put(chunk.getId(), chunk));
        System.out.println("Added " + chunksToAdd.size() + " chunks to in-memory vector store.");
    }

    @Override
    public List<DocumentChunk> similaritySearch(List<Float> queryEmbedding, int topK) {
        if (queryEmbedding == null || queryEmbedding.isEmpty()) {
            return Collections.emptyList();
        }

        List<SimilarityResult> results = new ArrayList<>();

        // Iterate through all stored chunks
        for (DocumentChunk chunk : chunks.values()) {
            if (chunk.getEmbedding() != null && !chunk.getEmbedding().isEmpty()) {
                double similarity = calculateCosineSimilarity(queryEmbedding, chunk.getEmbedding());
                results.add(new SimilarityResult(chunk, similarity));
            }
        }

        // Sort results by similarity in descending order
        results.sort(Comparator.comparingDouble(SimilarityResult::similarity).reversed());

        // Return topK chunks
        List<DocumentChunk> topChunks = new ArrayList<>();
        for (int i = 0; i < Math.min(topK, results.size()); i++) {
            topChunks.add(results.get(i).chunk());
        }
        return topChunks;
    }

    @Override
    public void clear() {
        chunks.clear();
        System.out.println("In-memory vector store cleared.");
    }

    /**
     * Calculates the cosine similarity between two vectors.
     * Cosine similarity measures the cosine of the angle between two vectors.
     * A value of 1 indicates identical direction, 0 indicates orthogonality, and -1 indicates opposite direction.
     *
     * @param vectorA The first vector.
     * @param vectorB The second vector.
     * @return The cosine similarity between the two vectors.
     * @throws IllegalArgumentException if vectors have different dimensions or are empty.
     */
    private double calculateCosineSimilarity(List<Float> vectorA, List<Float> vectorB) {
        if (vectorA.size() != vectorB.size()) {
            throw new IllegalArgumentException("Vectors must have the same dimensions.");
        }
        if (vectorA.isEmpty()) {
            throw new IllegalArgumentException("Vectors cannot be empty.");
        }

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < vectorA.size(); i++) {
            dotProduct += vectorA.get(i) * vectorB.get(i);
            normA += Math.pow(vectorA.get(i), 2);
            normB += Math.pow(vectorB.get(i), 2);
        }

        if (normA == 0.0 || normB == 0.0) {
            return 0.0; // Avoid division by zero if a vector is zero vector
        }

        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Helper class to store a DocumentChunk along with its calculated similarity score.
     */
    private record SimilarityResult(DocumentChunk chunk, double similarity) {
    }
}
