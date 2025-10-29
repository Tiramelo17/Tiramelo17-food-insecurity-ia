package br.com.food.insecutiry.model.dto;

import lombok.Builder;

import java.util.Map;

@Builder(toBuilder = true)
public record PredictionResponseDTO(String prediction, Map<String, Double> probabilities, String topAttributeName, Double topAttributeScore,  Map<String, Double> attributeImportances) {
}
