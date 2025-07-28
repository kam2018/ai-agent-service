package com.ai.agent.storage;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@NoArgsConstructor
@AllArgsConstructor
@Data
public class DocumentChunk {
    private String id;
    private String text;
    private List<Float> embedding;
    private String source;
}
