package br.com.food.insecutiry.repository;

import br.com.food.insecutiry.model.DatasetMetadata;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface DatasetMetadataRepository extends MongoRepository<DatasetMetadata, String> {
  DatasetMetadata findTopByOrderByIdDesc();
}
