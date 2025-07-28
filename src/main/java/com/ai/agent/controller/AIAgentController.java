package com.ai.agent.controller;

import com.ai.agent.model.Animal;
import com.ai.agent.model.Person;
import com.ai.agent.service.GeminiService;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/agent")
public class AIAgentController {

    private final GeminiService geminiService;

    public AIAgentController(GeminiService geminiService) {
        this.geminiService = geminiService;
    }

    @GetMapping("/persons")
    List<Person> generatePersonEntity() {
        PromptTemplate pt = new PromptTemplate("""
                Return a current list of 10 famous persons if exists or generate a new list with random values.
                Each object should contain an auto-incremented id field.
                Do not include any explanations or additional text.
                """);

        return geminiService.getResponse(pt, new ParameterizedTypeReference<List<Person>>() {});
    }

    @GetMapping("/animals")
    List<Animal> generateAnimalEntity() {
        PromptTemplate pt = new PromptTemplate("""
                Return a current list of 10 famous Animal if exists or generate a new list with random values.
                Each object should contain an auto-incremented id field.
                Do not include any explanations or additional text.
                """);

        return geminiService.getResponse(pt, new ParameterizedTypeReference<List<Animal>>() {});
    }

}
