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
O relatório do trabalho encontra-se em formato PDF dentro da pasta Trab1. 

# Trab 2
Em breve...
