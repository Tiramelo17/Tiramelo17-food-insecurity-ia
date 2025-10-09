package br.com.food.insecutiry.service;


import br.com.food.insecutiry.config.WekaModelConfig;
import br.com.food.insecutiry.model.DatasetMetadata;
import br.com.food.insecutiry.model.RangeAgeEstrategy;
import br.com.food.insecutiry.model.dto.DatasetInfoDTO;
import br.com.food.insecutiry.model.dto.PredictionRequestDTO;
import br.com.food.insecutiry.model.dto.PredictionResponseDTO;
import br.com.food.insecutiry.repository.DatasetMetadataRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.bson.types.ObjectId;
import org.springframework.data.mongodb.gridfs.GridFsTemplate;
import org.springframework.data.mongodb.gridfs.GridFsResource;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.*;
import java.util.Objects;
import java.util.Optional;


@RequiredArgsConstructor
@Slf4j
@Service
public class FoodInsecurityService { private final DatasetMetadataRepository datasetMetadataRepository;
  private final GridFsTemplate gridFsTemplate;
  private final WekaModelConfig wekaModelConfig;

  public PredictionResponseDTO predict(PredictionRequestDTO req) throws Exception {
    Classifier model = wekaModelConfig.getModel();
    if (model == null) {
      throw new IllegalStateException("Prediction model is not available. Please upload a dataset and train the model first.");
    }

    DatasetMetadata datasetMeta = datasetMetadataRepository.findTopByOrderByIdDesc();
    if (datasetMeta == null) throw new IllegalStateException("No dataset available.");

    GridFsResource resource = gridFsTemplate.getResource(datasetMeta.getFilename());
    File tempArff = File.createTempFile("structure", ".arff");
    try (
      InputStream is = resource.getInputStream();
      FileOutputStream fos = new FileOutputStream(tempArff)
    ) {
      is.transferTo(fos);
    }
    DataSource source = new DataSource(tempArff.getAbsolutePath());
    Instances structure = source.getDataSet();
    structure.setClassIndex(structure.numAttributes() - 1);

    String faixaIdade = RangeAgeEstrategy.fromAge(
      Optional.ofNullable(req.age())
        .orElseGet(() -> {
          if (req.ageRangeStart() != null && req.ageRangeEnd() != null) {
            return (int) ((req.ageRangeStart() + req.ageRangeEnd()) / 2.0);
          }
          return null;
        })
    );

    Instance inst = new DenseInstance(structure.numAttributes());
    inst.setDataset(structure);
    inst.setValue(0, req.salary());
    inst.setValue(1, req.region());
    inst.setValue(2, faixaIdade);
    inst.setValue(3, req.educationLevel());
    inst.setValue(4, req.country());

    double result = model.classifyInstance(inst);
    String predictedClass = structure.classAttribute().value((int) result);

    tempArff.delete();
    return new PredictionResponseDTO(predictedClass);
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
}
