package com.ai.agent.storage;

import java.util.List;

public interface VectorStore {
    void addDocumentChunks(List<DocumentChunk> chunks);

    List<DocumentChunk> similaritySearch(List<Float> queryEmbedding, int topK);

    void clear();
}
