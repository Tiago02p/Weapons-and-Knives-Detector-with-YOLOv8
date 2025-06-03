# AK-47 Detection with YOLOv8

Este projeto implementa um sistema de detecção de armas AK-47 usando YOLOv8, uma das mais recentes e eficientes arquiteturas de detecção de objetos. O sistema pode ser usado tanto para detecção em tempo real via webcam quanto para análise de imagens estáticas.

## 📋 Características

- Detecção de AK-47 em tempo real via webcam
- Análise de imagens estáticas
- Avaliação detalhada do modelo
- Métricas de performance (mAP, precisão, recall)
- Visualizações e gráficos de resultados
- Suporte para múltiplos formatos de imagem

## 🔄 Processo de Desenvolvimento

### 1. Coleta e Preparação dos Dados
- Criação de um dataset específico para AK-47
- Anotação manual das imagens para treinamento
- Divisão do dataset em conjuntos de treino e validação
- Pré-processamento das imagens para melhor performance

### 2. Treinamento do Modelo
- Utilização do YOLOv8 como base
- Fine-tuning com o dataset personalizado
- Ajuste de hiperparâmetros para otimização
- Múltiplas iterações de treinamento para melhorar a acurácia

### 3. Desenvolvimento do Sistema
- Implementação da detecção em tempo real
- Criação de scripts de avaliação
- Desenvolvimento de visualizações e métricas
- Otimização para performance em CPU

### 🎯 Desafios Enfrentados

1. **Dataset e Anotação**
   - Dificuldade em obter imagens variadas de AK-47
   - Processo demorado de anotação manual
   - Necessidade de balancear o dataset

2. **Treinamento**
   - Ajuste fino dos hiperparâmetros
   - Otimização do tempo de treinamento
   - Balanceamento entre acurácia e velocidade

3. **Implementação**
   - Otimização para CPU (sem GPU)
   - Ajuste do sistema de tracking em tempo real
   - Melhoria da performance em diferentes condições de iluminação

4. **Avaliação**
   - Desenvolvimento de métricas significativas
   - Criação de visualizações úteis
   - Testes em diferentes cenários

### 💡 Soluções Implementadas

1. **Para o Dataset**
   - Uso de técnicas de data augmentation
   - Criação de um pipeline de pré-processamento
   - Implementação de validação cruzada

2. **Para o Treinamento**
   - Implementação de early stopping
   - Uso de learning rate scheduling
   - Otimização de batch size

3. **Para a Performance**
   - Implementação de multi-threading
   - Otimização do processamento de imagens
   - Ajuste do sistema de tracking

4. **Para a Avaliação**
   - Desenvolvimento de scripts automatizados
   - Criação de visualizações interativas
   - Implementação de métricas detalhadas

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITÓRIO]
cd Weapons-and-Knives-Detector-with-YOLOv8
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 📦 Estrutura do Projeto

```
├── custom_dataset/           # Dataset de treinamento
│   ├── images/              # Imagens do dataset
│   └── labels/              # Anotações do dataset
├── evaluation_results/      # Resultados das avaliações
├── imgs/                    # Imagens para teste
├── runs/                    # Resultados do treinamento
├── models/                  # Modelos salvos
├── real_time_detection.py   # Detecção em tempo real
├── evaluate_model.py        # Avaliação do modelo
├── evaluate_custom_images.py # Avaliação em imagens personalizadas
└── requirements.txt         # Dependências do projeto
```

## 🎯 Uso

### 1. Detecção em Tempo Real

Para iniciar a detecção em tempo real usando sua webcam:

```bash
python real_time_detection.py
```

- Pressione 'q' para sair
- O sistema mostrará as detecções em tempo real
- FPS será exibido no canto superior esquerdo

### 2. Avaliação do Modelo

Para avaliar o desempenho geral do modelo:

```bash
python evaluate_model.py
```

Isso irá gerar:
- Métricas detalhadas (mAP, precisão, recall)
- Matriz de confusão
- Gráficos de performance
- Resultados salvos em `evaluation_results/`

### 3. Avaliação em Imagens Personalizadas

Para testar o modelo em um conjunto específico de imagens:

1. Coloque suas imagens na pasta `imgs/input/`
2. Execute:
```bash
python evaluate_custom_images.py
```

Resultados serão salvos em `evaluation_results/custom_images/`:
- Imagens anotadas com detecções
- CSV com resultados detalhados
- Gráficos de resumo
- Estatísticas de detecção

## 📊 Métricas de Performance

O modelo atual apresenta excelente performance:
- mAP50: 99.5%
- mAP50-95: 96.1%
- Precisão: 100%
- Recall: 100%

## 🔧 Configuração

### Ajustando o Limiar de Confiança

O limiar de confiança padrão é 0.5 (50%). Para ajustar:

1. Em `real_time_detection.py`:
```python
results = model.track(frame, persist=True, tracker="botsort.yaml", conf=0.5)  # Ajuste o valor 0.5
```

2. Em `evaluate_custom_images.py`:
```python
results = model(img, conf=0.5)  # Ajuste o valor 0.5
```

### Formatos de Imagem Suportados

- JPG/JPEG
- PNG
- BMP

## 📝 Notas

- O modelo foi treinado especificamente para detecção de AK-47
- Performance pode variar dependendo das condições de iluminação e ângulo
- Recomenda-se boa iluminação para melhores resultados

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor, sinta-se à vontade para submeter pull requests.

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 🙏 Créditos

- YOLOv8 por Ultralytics
- Dataset de treinamento personalizado
- Comunidade open source
