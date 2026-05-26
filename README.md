# stellar-classification-pytorch

Este repositório contém a implementação prática de uma Rede Neural Artificial desenvolvida em **PyTorch** para a classificação de objetos astronômicos (Galáxias, Quasares e Estrelas). 

## Dataset
Os experimentos utilizam o **Stellar Classification Dataset - SDSS17**, que conta com medições espectrais capturadas pelo *Sloan Digital Sky Survey*. O objetivo principal é predizer a classe do objeto com base em suas características fotométricas (bandas de filtros `u`, `g`, `r`, `i`, `z`) e no seu `redshift`.

---

## Arquitetura e Recursos Implementados
A arquitetura foi construída de forma flexível utilizando as capacidades do PyTorch, permitindo testar diferentes dinâmicas de treinamento:

- **Pré-processamento:** Tratamento de dados via Pandas e normalização de features utilizando o `StandardScaler` do Scikit-Learn.
- **Estrutura Modular:** Rede inteiramente baseada em `nn.Module` e `nn.Sequential`.
- **Função de Ativação:** Uso de **ReLU** nas camadas ocultas para introduzir não-linearidades.
- **Regularização:** Implementação de camadas de **Dropout** e penalização por **Weight Decay (Regularização L2)** para combater o Overfitting.
- **Otimização:** Uso do algoritmo **Adam** para a atualização dos pesos via Backpropagation.
