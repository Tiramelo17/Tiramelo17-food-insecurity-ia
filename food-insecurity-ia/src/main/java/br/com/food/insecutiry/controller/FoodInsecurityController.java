package br.com.food.insecutiry.controller;

import br.com.food.insecutiry.model.dto.DatasetInfoDTO;
import br.com.food.insecutiry.model.dto.PredictionRequestDTO;
import br.com.food.insecutiry.model.dto.PredictionResponseDTO;
import br.com.food.insecutiry.service.FoodInsecurityService;
import br.com.food.insecutiry.service.LocalFileIngestService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/food-insecurity")
@RequiredArgsConstructor
public class FoodInsecurityController {

  private final FoodInsecurityService service;
  private final LocalFileIngestService localFileIngestService;

  @PostMapping("/predict")
  public ResponseEntity<PredictionResponseDTO> predict(@RequestBody PredictionRequestDTO dto) throws Exception {
    return ResponseEntity.ok(service.predict(dto));
  }

  @PostMapping("/upload-dataset")
  public ResponseEntity<DatasetInfoDTO> uploadDataset(@RequestParam("file") MultipartFile file) throws Exception {
    DatasetInfoDTO info = service.uploadDataset(file);
    return ResponseEntity.accepted().body(info);
  }

  @PostMapping("/ingest-local-file")
  public void ingestLocalFile(@RequestParam("filePath") String filePath) throws Exception {
    localFileIngestService.saveLocalFileToDb(filePath);
  }
}