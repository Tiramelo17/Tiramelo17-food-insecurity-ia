package br.com.food.insecutiry.model;

import lombok.*;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "datasets_metadata")
public class DatasetMetadata {
  @Id
  private String id;
  private String filename;
  private String gridFsId; // reference to GridFS file
}
