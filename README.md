# Food Insecurity IA

Projeto de exemplo que utiliza Weka (biblioteca de aprendizado de máquina) em Java e Python para estudar e treinar modelos voltados para insegurança alimentar. O repositório combina código Java (uso principal do Weka) e scripts Python de apoio / experimentação. A pasta de dados utilizada para treino e testes inclui arquivos no formato ARFF — a base de teste principal é `base_granular_para_ml_usadas_para_testes.arff`.

Sumário
- Visão geral
- Pré-requisitos
- Estrutura do projeto
- Uso do Weka com Java
- Exemplo de classe Config para treinar IA
- Como executar
- Evidência (dataset usado)
- Licença

## Visão geral

Este repositório demonstra um pipeline simples de ML usando Weka em Java:
- carregar dados ARFF,
- configurar um classificador (ex.: RandomForest, J48, etc.),
- treinar o modelo via `buildClassifier`,
- avaliar com validação cruzada,
- persistir o modelo treinado.

Há também código Python para manipulação de dados e experimentos complementares.

## Pré-requisitos

- Java 21 (JDK)
- SpringBoot 3.2.6
- IDE (opcional): IntelliJ
- Weka (adicione `weka.jar` ao classpath) — ou adicione a dependência via Maven/Gradle se preferir
  - Alternativa: baixar o weka.jar em https://www.cs.waikato.ac.nz/ml/weka/
- Arquivo ARFF de entrada (ex.: `base_granular_para_ml_usadas_para_testes.arff`)
- (Opcional) Python 3.x para scripts auxiliares

Observação sobre dependências Java:
- Se não for usar `weka.jar` manualmente, adicione a dependência em seu build system (Maven/Gradle) ou inclua o jar no classpath da aplicação.

## Estrutura do projeto (exemplo)
- /src/java/... — código Java que usa Weka
- /src/python/... — scripts auxiliares em Python
- /data — arquivos ARFF e outros dados
  - base_granular_para_ml_usadas_para_testes.arff
- /models — modelos serializados (após treino)
- README.md

## Uso do Weka com Java (resumo rápido)

1. Carregar os dados:
   - Use `weka.core.converters.ConverterUtils.DataSource` para carregar ARFF/CSV.
   - Ajuste o índice da classe com `instances.setClassIndex(...)`.

2. Criar e configurar o classificador:
   - Ex.: `weka.classifiers.trees.RandomForest`, `weka.classifiers.trees.J48`, `weka.classifiers.functions.SMO`, etc.
   - Configure opções com `weka.core.Utils.splitOptions(...)` ou via setters disponíveis.

3. Treinar:
   - `classifier.buildClassifier(instances)`

4. Avaliar:
   - `weka.classifiers.Evaluation` e usar `crossValidateModel` ou `evaluateModel`.

5. Salvar/Carregar modelo:
   - `weka.core.SerializationHelper.write("modelo.model", classifier)`
   - `Classifier cls = (Classifier) SerializationHelper.read("modelo.model")`

Exemplo de imports úteis:
- weka.core.converters.ConverterUtils.DataSource
- weka.core.Instances
- weka.classifiers.Classifier
- weka.classifiers.trees.RandomForest
- weka.classifiers.Evaluation
- weka.core.SerializationHelper
- weka.core.Utils

## Exemplo de classe Config para treinar IA (Java)

Abaixo está um exemplo ilustrativo de como uma classe `TrainingConfig` pode centralizar parâmetros e executar o treino. Ajuste conforme estrutura do seu projeto.

```java
// Exemplo simplificado de uso do Weka em Java
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;
import weka.core.Utils;

import java.util.Random;

public class TrainingConfig {
    private String arffPath;
    private String classifierOptions; // ex: "-I 100 -K 0 -S 1" para RandomForest
    private String modelOutputPath;
    private int folds = 10;
    private int seed = 1;

    public TrainingConfig(String arffPath, String classifierOptions, String modelOutputPath) {
        this.arffPath = arffPath;
        this.classifierOptions = classifierOptions;
        this.modelOutputPath = modelOutputPath;
    }

    public void trainAndEvaluate() throws Exception {
        // 1. Carregar dados
        DataSource source = new DataSource(arffPath);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1); // considerar última coluna como classe
        }

        // 2. Configurar classificador
        RandomForest rf = new RandomForest();
        if (classifierOptions != null && !classifierOptions.trim().isEmpty()) {
            rf.setOptions(Utils.splitOptions(classifierOptions));
        }

        // 3. Treinar
        rf.buildClassifier(data);

        // 4. Avaliar com validação cruzada
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(rf, data, folds, new Random(seed));

        System.out.println(eval.toSummaryString("=== Resumo ===", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());

        // 5. Salvar modelo
        SerializationHelper.write(modelOutputPath, rf);
        System.out.println("Modelo salvo em: " + modelOutputPath);
    }

    // getters / setters omitidos para brevidade
}
```

Uso rápido (classe com main):

```java
public class TrainMain {
    public static void main(String[] args) throws Exception {
        String arff = "data/base_granular_para_ml_usadas_para_testes.arff";
        String options = "-I 100 -K 0 -S 1"; // exemplo para RandomForest
        String modelOut = "models/randomforest.model";

        TrainingConfig cfg = new TrainingConfig(arff, options, modelOut);
        cfg.trainAndEvaluate();
    }
}
```

Notas:
- Ajuste `setClassIndex` se a classe não for a última coluna.
- Para conjuntos de dados muito grandes, considere amostragem, seleção de atributos e pré-processamento.
- Para outros classificadores substitua `RandomForest` pelo classificador desejado e ajuste opções.

## Como executar

1. Garanta que `weka.jar` esteja no classpath:
   - Exemplo (linha de comando):
     java -cp "lib/weka.jar:out/production/SeuProjeto:." TrainMain
   - Ou configure via Maven/Gradle.

2. Execute a classe `TrainMain` (ou equivalente) que instancia `TrainingConfig`.

3. Verifique a saída no console (métricas) e confirme que o modelo foi salvo em `/models`.

## Evidência — dados usados para treinamento

Os dados usados para treinamento e teste deste projeto estão disponíveis neste repositório público no Kaggle. A base usada para teste tem o nome:

- base usada para teste: `base_granular_para_ml_usadas_para_testes.arff`

Link para as bases utilizadas (evidência):
https://www.kaggle.com/datasets/tiramelo17/food-insecurity-ia?select=base_granular_para_ml_usadas_para_testes.arff

> Acesse o link acima para baixar a(s) base(s) ARFF usadas nos experimentos.

## Contribuição

Contribuições são bem-vindas. Abra issues para bug reports e pull requests para melhorias. Indique no PR quais alterações foram feitas e exemplos de execução.

## Licença

Coloque aqui a licença do projeto (ex.: MIT) ou outra desejada. Se não houver ainda, adicione um arquivo LICENSE apropriado.

---

Se desejar, eu posso:
- gerar o arquivo README.md no repositório,
- adicionar um exemplo de pom.xml com dependência do Weka,
- escrever um script de treinamento completo (classe Main) baseado nas suas pastas atuais.
