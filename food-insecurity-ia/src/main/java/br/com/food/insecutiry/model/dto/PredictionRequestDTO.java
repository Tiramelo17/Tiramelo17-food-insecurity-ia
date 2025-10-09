package br.com.food.insecutiry.model.dto;

import lombok.Builder;

@Builder(toBuilder = true)
public record PredictionRequestDTO(
  Double salary,
  String region,
  Integer age,
  Integer ageRangeStart,
  Integer ageRangeEnd,
  String educationLevel,
  String country ) {
}
