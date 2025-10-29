package br.com.food.insecutiry.service;


import br.com.food.insecutiry.config.WekaModelConfig;
import br.com.food.insecutiry.model.DatasetMetadata;
import br.com.food.insecutiry.model.dto.DatasetInfoDTO;
import br.com.food.insecutiry.model.dto.PredictionRequestDTO;
import br.com.food.insecutiry.model.dto.PredictionResponseDTO;
import br.com.food.insecutiry.repository.DatasetMetadataRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.bson.types.ObjectId;
import org.springframework.data.mongodb.gridfs.GridFsTemplate;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.*;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;


@RequiredArgsConstructor
@Slf4j
@Service
public class FoodInsecurityService {
  private final DatasetMetadataRepository datasetMetadataRepository;
  private final GridFsTemplate gridFsTemplate;
  private final WekaModelConfig wekaModelConfig;

  public PredictionResponseDTO predict(PredictionRequestDTO req) throws Exception {
    Classifier model = wekaModelConfig.getModel();
    if (model == null) {
      throw new IllegalStateException("Prediction model is not available. Please upload a dataset and train the model first.");
    }

    Instances structure = wekaModelConfig.getStructure();
    if (structure == null) {
      throw new IllegalStateException("Dataset structure not available.");
    }

    // cria instância
    Instance inst = new DenseInstance(structure.numAttributes());
    inst.setDataset(new Instances(structure)); // usar cópia leve para thread-safety
    String salarioRange = mapSalaryToRange(req.salario());
    if (salarioRange == null) inst.setMissing(0); else inst.setValue(0, salarioRange);
    inst.setValue(1, req.estado());
    inst.setValue(2, req.cor());
    inst.setValue(3, req.escolaridade());

    // obtém distribuição de probabilidades por classe (ex: [0.9, 0.1])
    double[] dist = model.distributionForInstance(inst);

    // mapeia para rótulos e percentuais
    Map<String, Double> percentages = new LinkedHashMap<>();
    for (int i = 0; i < dist.length; i++) {
      String label = structure.classAttribute().value(i);
      double pct = dist[i] * 100.0;
      percentages.put(label, pct);
    }

    // classe prevista (a com maior probabilidade)
    int bestIdx = 0;
    for (int i = 1; i < dist.length; i++) if (dist[i] > dist[bestIdx]) bestIdx = i;
    String predictedClass = structure.classAttribute().value(bestIdx);

    Optional<Map.Entry<String, Double>> topAttr = wekaModelConfig.getTopAttribute();

    return PredictionResponseDTO.builder()
      .probabilities(percentages)
      .prediction(predictedClass)
      .topAttributeName(topAttr.map(Map.Entry::getKey).orElse(null))
      .topAttributeScore(topAttr.map(Map.Entry::getValue).orElse(null))
      .attributeImportances(wekaModelConfig.getAttributeImportances())
      .build();
  }

  public DatasetInfoDTO uploadDataset(MultipartFile file) throws IOException {
    ObjectId fileId = gridFsTemplate.store(file.getInputStream(), Objects.requireNonNull(file.getOriginalFilename()));
    DatasetMetadata metadata = DatasetMetadata.builder()
      .filename(file.getOriginalFilename())
      .gridFsId(fileId.toHexString())
      .build();
    datasetMetadataRepository.save(metadata);

    wekaModelConfig.trainModelFromDataset(metadata);

    return new DatasetInfoDTO(metadata.getId(), metadata.getFilename());
  }

  private String mapSalaryToRange(Double salary) {
    if (salary == null) return null;
    double s = salary;
    if (s >= 422 && s <= 999) return "422-999";
    if (s >= 810 && s <= 1300) return "810-1300";
    if (s >= 1500 && s <= 3000) return "1500-3000";
    if (s >= 0 && s <= 499) return "0-499";
    if (s >= 500 && s <= 999) return "500-999";
    if (s >= 1000 && s <= 1499) return "1000-1499";
    if (s >= 1500 && s <= 1999) return "1500-1999";
    if (s >= 2000 && s <= 2499) return "2000-2499";
    if (s >= 2500 && s <= 2999) return "2500-2999";
    return "3000-50000";
  }
}
