package br.com.food.insecutiry.model.dto;

import lombok.Builder;

@Builder(toBuilder = true)
public record PredictionResponseDTO(String prediction) {
}
