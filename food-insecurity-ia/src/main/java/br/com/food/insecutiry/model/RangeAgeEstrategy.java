package br.com.food.insecutiry.model;

import lombok.Getter;

import java.util.function.Predicate;

public enum RangeAgeEstrategy {
  FAIXA_0_17("0-17", age -> age != null && age <= 17),
  FAIXA_18_29("18-29", age -> age != null && age <= 29),
  FAIXA_30_44("30-44", age -> age != null && age <= 44),
  FAIXA_45_59("45-59", age -> age != null && age <= 59),
  FAIXA_60_MAIS("60+", age -> age != null && age > 59);

  @Getter
  private final String value;
  private final Predicate<Integer> predicate;

  RangeAgeEstrategy(String value, Predicate<Integer> predicate) {
    this.value = value;
    this.predicate = predicate;
  }

  public static String fromAge(Integer age) {
    for (RangeAgeEstrategy range : values()) {
      if (range.predicate.test(age)) {
        return range.getValue();
      }
    }
    return "0-17";
  }
}
