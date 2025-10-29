package br.com.food.insecutiry.service;

import br.com.food.insecutiry.config.WekaModelConfig;
import br.com.food.insecutiry.model.DatasetMetadata;
import br.com.food.insecutiry.repository.DatasetMetadataRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.mongodb.gridfs.GridFsTemplate;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;

@Service
@RequiredArgsConstructor
public class LocalFileIngestService {

  private final GridFsTemplate gridFsTemplate;
  private final DatasetMetadataRepository datasetMetadataRepository;
  private final WekaModelConfig wekaModelConfig;


  // salva arquivo local no GridFS e cria entrada em DatasetMetadata
  public void saveLocalFileToDb(String localPath) {
    try {
      File file = new File(localPath);
      if (!file.exists() || !file.isFile()) {
        throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Arquivo local não encontrado: " + localPath);
      }

      String filename = file.getName();
      try (InputStream is = new FileInputStream(file)) {
        gridFsTemplate.store(is, filename, "application/arff");
      }

      DatasetMetadata meta = new DatasetMetadata();
      meta.setFilename(filename);
      datasetMetadataRepository.save(meta);
      wekaModelConfig.trainModelFromDataset(meta);
    } catch (ResponseStatusException rse) {
      throw rse;
    } catch (Exception e) {
      throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Erro salvando arquivo local no DB: " + e.getMessage(), e);
    }
  }

  public DatasetMetadata findLatestDataset() {
    return datasetMetadataRepository.findTopByOrderByIdDesc();
  }
}