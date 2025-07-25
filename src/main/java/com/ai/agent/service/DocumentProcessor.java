package com.ai.agent.service;

import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
public class DocumentProcessor {
    private static final int CHUNK_SIZE = 500; // Max characters per chunk
    private static final int CHUNK_OVERLAP = 100; // Overlap between consecutive chunks

    public List<String> splitDocument(String documentText) {
        List<String> chunks = new ArrayList<>();
        if (documentText == null || documentText.isEmpty()) {
            return chunks;
        }

        long currentPosition = 0;
        long docLength = documentText.length();
        while (currentPosition < docLength) {
            long endPosition = Math.min(currentPosition + CHUNK_SIZE, documentText.length());
            String chunk = documentText.substring((int) currentPosition, (int) endPosition);
            chunks.add(chunk);

            // Move to the next chunk, applying overlap
            currentPosition += (CHUNK_SIZE - CHUNK_OVERLAP);
            if (currentPosition < 0) { // Handle case where chunk size is smaller than overlap
                currentPosition = 0;
            }
        }
        return chunks;
    }
}
