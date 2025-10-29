
package br.com.food.insecutiry.config;

import jakarta.annotation.PostConstruct;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import weka.core.converters.ConverterUtils.DataSource;
import br.com.food.insecutiry.model.DatasetMetadata;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.mongodb.gridfs.GridFsTemplate;
import org.springframework.data.mongodb.gridfs.GridFsResource;
import br.com.food.insecutiry.repository.DatasetMetadataRepository;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.attributeSelection.InfoGainAttributeEval;
import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Configuration
@RequiredArgsConstructor
public class WekaModelConfig {

  private static final String MODEL_PATH = "food_insecurity_model.model";
  private static final String STRUCTURE_PATH = "food_insecurity_structure.header";

  private final DatasetMetadataRepository datasetMetadataRepository;
  private final GridFsTemplate gridFsTemplate;

  @Getter
  private volatile Classifier model;

  @Getter
  private volatile Instances structure;

  @Getter
  private volatile Map<String, Double> attributeImportances = Collections.emptyMap();


  public Optional<Map.Entry<String, Double>> getTopAttribute() {
    return attributeImportances.entrySet().stream().max(Map.Entry.comparingByValue());
  }

  @PostConstruct
  public void init() {
    try {
      File modelFile = new File(MODEL_PATH);
      if (modelFile.exists()) {
        log.info("Loading Weka model from disk...");
        model = (Classifier) weka.core.SerializationHelper.read(MODEL_PATH);

        File structFile = new File(STRUCTURE_PATH);
        if (structFile.exists()) {
          try {
            Object obj = weka.core.SerializationHelper.read(STRUCTURE_PATH);
            if (obj instanceof Instances) {
              structure = (Instances) obj;
              log.info("Loaded dataset structure from disk.");
            } else {
              log.warn("Structure file exists but content is not Instances.");
            }
          } catch (Exception e) {
            log.warn("Failed to read structure file: {}. Will attempt to retrain if dataset available.", e.getMessage());
            structure = null;
          }
        } else {
          log.warn("Structure file not found on disk.");
        }

        if (structure == null) {
          DatasetMetadata datasetMeta = datasetMetadataRepository.findTopByOrderByIdDesc();
          if (datasetMeta != null) {
            log.info("Retraining model to recover structure...");
            trainModelFromDataset(datasetMeta);
            log.info("Retrain complete.");
          } else {
            log.warn("No dataset available to recover structure.");
          }
        }

      } else {
        log.info("Model file not found. Will train when dataset is uploaded and requested.");
        DatasetMetadata datasetMeta = datasetMetadataRepository.findTopByOrderByIdDesc();
        if (datasetMeta != null) {
          log.info("Found dataset in database. Training model...");
          trainModelFromDataset(datasetMeta);
          log.info("Model training complete.");
        } else {
          log.warn("No dataset available for training. Prediction will not be available until a dataset is uploaded.");
          model = null;
          structure = null;
        }
      }
    } catch (Exception e) {
      log.error("Error initializing Weka model: {}", e.getMessage(), e);
      model = null;
      structure = null;
      attributeImportances = Collections.emptyMap();
    }
  }

  public void trainModelFromDataset(DatasetMetadata datasetMeta) {
    try {
      GridFsResource resource = gridFsTemplate.getResource(datasetMeta.getFilename());
      File tempArff = File.createTempFile("dataset", ".arff");
      try (
        InputStream is = resource.getInputStream();
        FileOutputStream fos = new FileOutputStream(tempArff)
      ) {
        is.transferTo(fos);
      }

      DataSource source = new DataSource(tempArff.getAbsolutePath());
      Instances data = source.getDataSet();
      data.setClassIndex(data.numAttributes() - 1);

      // Treina o classificador J48 (árvore de decisão)
      Classifier newModel = new weka.classifiers.trees.J48();
      newModel.buildClassifier(data);

      // armazena modelo em disco
      weka.core.SerializationHelper.write(MODEL_PATH, newModel);
      this.model = newModel;

      // armazena apenas o header/estrutura (sem instâncias) em memória e em disco
      Instances header = new Instances(data, 0);
      this.structure = header;
      weka.core.SerializationHelper.write(STRUCTURE_PATH, header);

      // calcula importâncias via InfoGain
      try {
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        eval.buildEvaluator(data);
        Map<String, Double> map = new LinkedHashMap<>();
        for (int i = 0; i < data.numAttributes(); i++) {
          if (i == data.classIndex()) continue;
          String name = data.attribute(i).name();
          double score = eval.evaluateAttribute(i);
          map.put(name, score);
        }
        // ordena por score decrescente e guarda imutável
        this.attributeImportances = map.entrySet().stream()
          .sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
          .collect(Collectors.toMap(
            Map.Entry::getKey,
            Map.Entry::getValue,
            (a,b) -> a,
            LinkedHashMap::new
          ));
      } catch (Exception e) {
        log.warn("Failed computing attribute importances: {}", e.getMessage());
        this.attributeImportances = Collections.emptyMap();
      }

      tempArff.delete();
      log.info("Model and structure trained and saved successfully.");
    } catch (Exception e) {
      log.error("Error training Weka model: {}", e.getMessage(), e);
      model = null;
      structure = null;
      attributeImportances = Collections.emptyMap();
    }
  }

  public void setTrainedModel(Classifier trainedModel, Instances datasetStructure) {
    this.model = trainedModel;
    if (datasetStructure != null) {
      this.structure = new Instances(datasetStructure, 0);
      try {
        weka.core.SerializationHelper.write(STRUCTURE_PATH, this.structure);
      } catch (Exception e) {
        log.warn("Failed to persist structure to disk: {}", e.getMessage());
      }
    } else {
      this.structure = null;
    }
    if (trainedModel != null) {
      try {
        weka.core.SerializationHelper.write(MODEL_PATH, trainedModel);
      } catch (Exception e) {
        log.warn("Failed to persist model to disk: {}", e.getMessage());
      }
    }
  }
}
