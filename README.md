# Trabalhos de Tópicos Especiais em Arquitetura de Computadores
Neste respositório encontram-se os códigos utilizando GPU para resolução dos trabalhos da disciplina Tópicos Especiais em Arquitetura de Computadores da Universidade Federal do Espírito Santo

# Trab 1
Nesta pasta encontra-se o Trabalho 1 da disciplina. Os códigos estão divididos em duas pastas:
- float: código para uso da GPU para dados com precisão simples
- double: código para uso da GPU para dados com precisão dupla

Para compilar basta utilizar o makefile contido em cada pasta. Obviamente é necessário que CUDA esteja instalado na máquina. Para incluir o caminho do CUDA como variável de ambiente cada pasta contém o script cuda_path.sh:
```sh
export CUDA_HOME=/usr/local/cuda-8.0 
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 
PATH=${CUDA_HOME}/bin:${PATH} 
export PATH
```
Para determinar o número de threads utilizadas pela OpenMP, o script roda_tudo.sh inclui na primeira linha:
```sh
export OMP_NUM_THREADS=4
```
Além disso, esse script chama o executável com diferentes tamanhos de matrizes.
O relatório técnico deste trabalho pode ser acessado [Neste link](https://arxiv.org/abs/1905.03641)

**O relatório deste trabalho está disponível no ARXIV:**
COLOCAR

# Trab 2
Nesta pasta encontra-se o Trabalho 2 da disciplina. Código esta contido no arquivo Trab2.py. O código possui as seguintes depêndencias de bibliotecas do python:
- TensorFlow
- Keras
- Numpy
- cv2
- pandas

Além disso, por utilizar funções de SO para carregar a bases de dados, este código funcina apenas no linux. A base de dados original pode ser carregada a partir da página do desafio no [Kaggle](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data). As modificações realizadas nas bases e o teste de validaço podem ser obtidas [aqui](https://www.dropbox.com/s/hy9k55z41xb97ey/baseteste2.zip?dl=0).

Para utilizar o código basta alterar o Path das bases nos locais indicados no código. Os modelos também podem ser alterados via parâmetros, também indicados como comentários no código.

O relatório técnico deste trabalho pode ser acessado [Neste link](https://arxiv.org/abs/1905.03642)
