package com.ai.agent.model;

public record Person (Integer id, String firstName, String lastName, int age, Gender gender,
                      String nationality) {
}

