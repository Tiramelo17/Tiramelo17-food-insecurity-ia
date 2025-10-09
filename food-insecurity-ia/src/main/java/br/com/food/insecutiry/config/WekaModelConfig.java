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
import java.io.*;

@Slf4j
@Configuration
@RequiredArgsConstructor
public class WekaModelConfig {

  private static final String MODEL_PATH = "food_insecurity_model.model";

  private final DatasetMetadataRepository datasetMetadataRepository;
  private final GridFsTemplate gridFsTemplate;

  @Getter
  private volatile Classifier model;

  @PostConstruct
  public void init() {
    try {
      File modelFile = new File(MODEL_PATH);
      if (modelFile.exists()) {
        log.info("Loading Weka model from disk...");
        model = (Classifier) weka.core.SerializationHelper.read(MODEL_PATH);
      } else {
        log.info("Model file not found. Will train when dataset is uploaded and requested.");
        DatasetMetadata datasetMeta = datasetMetadataRepository.findTopByOrderByIdDesc();
        if (datasetMeta != null) {
          log.info("Found dataset in database. Training model...");
          trainModelFromDataset(datasetMeta);
        } else {
          log.warn("No dataset available for training. Prediction will not be available until a dataset is uploaded.");
          model = null;
        }
      }
    } catch (Exception e) {
      log.error("Error initializing Weka model: {}", e.getMessage(), e);
      model = null;
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

      // You can change the classifier here if needed.
      Classifier newModel = new weka.classifiers.trees.J48();
      newModel.buildClassifier(data);

      weka.core.SerializationHelper.write(MODEL_PATH, newModel);
      this.model = newModel;

      tempArff.delete();
      log.info("Model trained and saved successfully.");
    } catch (Exception e) {
      log.error("Error training Weka model: {}", e.getMessage(), e);
      model = null;
    }
  }
}