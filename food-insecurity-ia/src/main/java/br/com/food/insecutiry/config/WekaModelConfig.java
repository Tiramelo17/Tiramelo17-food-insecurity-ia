
package br.com.food.insecutiry.config;

import jakarta.annotation.PostConstruct;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import weka.core.Instance;
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

      log.info("=== Diagnósticos do Dataset ===");
      log.info("Total de instâncias: {}", data.numInstances());
      log.info("Total de atributos: {}", data.numAttributes());
      log.info("Atributo classe: {}", data.classAttribute().name());

      data.randomize(new Random(42));
      int trainSize = (int) Math.round(data.numInstances() * 0.7);
      int testSize = data.numInstances() - trainSize;

      Instances trainData = new Instances(data, 0, trainSize);
      Instances testData = new Instances(data, trainSize, testSize);

      log.info("=== Divisão dos Dados ===");
      log.info("Conjunto de treino: {} instâncias ({}%)", trainSize, 70.0);
      log.info("Conjunto de teste: {} instâncias ({}%)", testSize, 30.0);

      Classifier newModel = new weka.classifiers.trees.J48();
      long startTrain = System.currentTimeMillis();
      newModel.buildClassifier(trainData);
      long trainTime = System.currentTimeMillis() - startTrain;

      log.info("Tempo de treinamento: {} ms", trainTime);

      weka.classifiers.Evaluation evalTest = new weka.classifiers.Evaluation(trainData);
      try {
        File csv = new File("calibration_test.csv");
        exportCalibrationCsv(testData, newModel, csv);
        log.info("Calibration CSV gravado em: {}", csv.getAbsolutePath());
      } catch (Exception e) {
        log.warn("Falha ao exportar CSV de calibragem: {}", e.getMessage());
      }


      long startTest = System.currentTimeMillis();
      evalTest.evaluateModel(newModel, testData);
      long testTime = System.currentTimeMillis() - startTest;

      log.info("=== Métricas no Conjunto de Teste (Holdout 70/30) ===");
      log.info("Tempo de teste: {} ms", testTime);
      log.info("Acurácia: {}%", evalTest.pctCorrect());
      log.info("Instâncias corretas: {}", evalTest.correct());
      log.info("Instâncias incorretas: {}", evalTest.incorrect());
      log.info("Kappa statistic: {}", evalTest.kappa());
      log.info("Erro médio absoluto: {}", evalTest.meanAbsoluteError());
      log.info("Raiz do erro quadrático médio: {}", evalTest.rootMeanSquaredError());

      // Métricas por classe (TESTE)
      for (int i = 0; i < trainData.numClasses(); i++) {
        log.info("=== Classe: {} (Teste) ===", trainData.classAttribute().value(i));
        log.info("Precision: {}", evalTest.precision(i));
        log.info("Recall: {}", evalTest.recall(i));
        log.info("F-Measure: {}", evalTest.fMeasure(i));
        try {
          log.info("AUC: {}", evalTest.areaUnderROC(i));
        } catch (Exception e) {
          log.warn("AUC não disponível para classe {}", i);
        }
      }

      log.info("=== Matriz de Confusão (Teste) ===");
      log.info("\n{}", evalTest.toMatrixString());

      log.info("=== Validação Cruzada 10-Fold (apenas no treino) ===");
      weka.classifiers.Evaluation evalCV = new weka.classifiers.Evaluation(trainData);
      Classifier modelCV = new weka.classifiers.trees.J48();
      evalCV.crossValidateModel(modelCV, trainData, 10, new Random(42));

      log.info("Acurácia (CV no treino): {}%", evalCV.pctCorrect());
      log.info("Kappa (CV no treino): {}", evalCV.kappa());

      // === SALVAR MODELO FINAL (treinado no conjunto COMPLETO para produção) ===
      log.info("=== Treinando modelo final no dataset completo ===");
      Classifier finalModel = new weka.classifiers.trees.J48();
      finalModel.buildClassifier(data);
      weka.core.SerializationHelper.write(MODEL_PATH, finalModel);
      this.model = finalModel;

      Instances header = new Instances(data, 0);
      this.structure = header;
      weka.core.SerializationHelper.write(STRUCTURE_PATH, header);

      try {
        InfoGainAttributeEval evalAttr = new InfoGainAttributeEval();
        evalAttr.buildEvaluator(data);
        Map<String, Double> map = new LinkedHashMap<>();
        for (int i = 0; i < data.numAttributes(); i++) {
          if (i == data.classIndex()) continue;
          String name = data.attribute(i).name();
          double score = evalAttr.evaluateAttribute(i);
          map.put(name, score);
        }
        this.attributeImportances = map.entrySet().stream()
          .sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
          .collect(Collectors.toMap(
            Map.Entry::getKey,
            Map.Entry::getValue,
            (a,b) -> a,
            LinkedHashMap::new
          ));

        log.info("=== Top 5 Atributos Mais Importantes ===");
        attributeImportances.entrySet().stream()
          .limit(5)
          .forEach(e -> log.info("{}: {}", e.getKey(), e.getValue()));

      } catch (Exception e) {
        log.warn("Failed computing attribute importances: {}", e.getMessage());
        this.attributeImportances = Collections.emptyMap();
      }

      testDataLeakageWithShuffledLabels(datasetMeta);

      tempArff.delete();
      log.info("Model and structure trained and saved successfully.");
    } catch (Exception e) {
      log.error("Error training Weka model: {}", e.getMessage(), e);
      model = null;
      structure = null;
      attributeImportances = Collections.emptyMap();
    }
  }

  public void testDataLeakageWithShuffledLabels(DatasetMetadata datasetMeta) {
    try {
      GridFsResource resource = gridFsTemplate.getResource(datasetMeta.getFilename());
      File tempArff = File.createTempFile("dataset_shuffled", ".arff");
      try (
        InputStream is = resource.getInputStream();
        FileOutputStream fos = new FileOutputStream(tempArff)
      ) {
        is.transferTo(fos);
      }

      DataSource source = new DataSource(tempArff.getAbsolutePath());
      Instances data = source.getDataSet();
      data.setClassIndex(data.numAttributes() - 1);

      log.info("=== TESTE DE VAZAMENTO: Embaralhando Rótulos ===");

      Random rand = new Random(42);
      double[] shuffledLabels = new double[data.numInstances()];
      for (int i = 0; i < data.numInstances(); i++) {
        shuffledLabels[i] = data.instance(i).classValue();
      }

      // Fisher-Yates shuffle
      for (int i = shuffledLabels.length - 1; i > 0; i--) {
        int j = rand.nextInt(i + 1);
        double temp = shuffledLabels[i];
        shuffledLabels[i] = shuffledLabels[j];
        shuffledLabels[j] = temp;
      }

      for (int i = 0; i < data.numInstances(); i++) {
        data.instance(i).setClassValue(shuffledLabels[i]);
      }

      data.randomize(new Random(42));
      int trainSize = (int) Math.round(data.numInstances() * 0.7);
      Instances trainData = new Instances(data, 0, trainSize);
      Instances testData = new Instances(data, trainSize, data.numInstances() - trainSize);

      Classifier testModel = new weka.classifiers.trees.J48();
      testModel.buildClassifier(trainData);

      weka.classifiers.Evaluation eval = new weka.classifiers.Evaluation(trainData);
      eval.evaluateModel(testModel, testData);

      log.info("=== RESULTADOS COM RÓTULOS EMBARALHADOS ===");
      log.info("Acurácia: {}%", eval.pctCorrect());
      log.info("Kappa: {}", eval.kappa());
      double expectedAccuracy = 100.0 / data.numClasses();
      log.info("Acurácia esperada com rótulos aleatórios: ~{}%", expectedAccuracy);

      if (eval.pctCorrect() > expectedAccuracy + 20) {
        log.warn("⚠️ ALERTA: Acurácia muito alta com rótulos embaralhados!");
        log.warn("⚠️ Forte indício de VAZAMENTO DE DADOS no dataset!");
      } else {
        log.info("✓ Teste de vazamento OK: acurácia dentro do esperado para rótulos aleatórios.");
      }

      tempArff.delete();
    } catch (Exception e) {
      log.error("Erro no teste de vazamento: {}", e.getMessage(), e);
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

  public void exportCalibrationCsv(Instances testData, Classifier classifier, File outFile) throws IOException {
    try (BufferedWriter bw = new BufferedWriter(new FileWriter(outFile))) {
      // Cabeçalho
      StringBuilder header = new StringBuilder();
      header.append("instance_index,true_label");
      for (int c = 0; c < testData.numClasses(); c++) {
        String clsName = testData.classAttribute().value(c).replaceAll("[,\\n\\r]", "_");
        header.append(",prob_class_").append(clsName);
      }
      bw.write(header.toString());
      bw.newLine();

      // Linhas por instância
      for (int i = 0; i < testData.numInstances(); i++) {
        Instance inst = testData.instance(i);
        double[] probs;
        try {
          probs = classifier.distributionForInstance(inst);
        } catch (Exception e) {
          // se distribution falhar, escreve zeros
          probs = new double[testData.numClasses()];
        }
        String trueLabel = testData.classAttribute().value((int) inst.classValue());
        StringBuilder line = new StringBuilder();
        line.append(i).append(",").append(trueLabel);
        for (double p : probs) {
          line.append(",").append(p);
        }
        bw.write(line.toString());
        bw.newLine();
      }
    }
  }
}
