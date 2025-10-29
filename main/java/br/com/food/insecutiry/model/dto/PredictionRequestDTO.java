package br.com.food.insecutiry.model.dto;

import lombok.Builder;

@Builder(toBuilder = true)
public record PredictionRequestDTO(
  Double salario,
  String estado,
  String cor,
  Integer escolaridade,
  Boolean inseguranca) {
}
