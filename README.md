# Zeroth Order Policy Search Methods for Global Optimization Problems: An Experimental Study
Os métodos Policy Search (PS) vem sendo utilizados nos últimos anos para se aprender, automaticamente, algoritmos de otimização, obtendo resultados animadores. Nesse repositório, disponibilizamos os códigos utilizados para a comparação de 4 algoritmos dessa família de métodos (REINFORCE, SAC, TD3 e PPO) para resolver 8 problemas de otimização global, aprendendo diferentes algoritmos de otimização de ordem zero.

## Sumário

1. <a href='#Introduction'>Sobre o Repositório</a><br>
2. <a href='#BenchmarkFunctions'>Funções de Benchmark</a><br>
3. <a href='#ConvergencePlots'>Gráfico de Convergência</a><br>
4. <a href='#Comparisons'>Comparação em diferentes dimensões</a><br>

<a id='Introduction'></a>
## Sobre o Repositório
Esse repositório contém os códigos utilizados pelo artigo **_Zeroth Order Policy Search Methods for Global Optimization Problems: An Experimental Study_**, submetido ao [_ENIAC 2021_](http://c4ai.inova.usp.br/bracis/eniac.htm). Encorajamos ao leitor realizar testes com os códigos e agentes disponibilizados.

### Estrutura do Repositório
```
.
├── imgs
│
├── policies
|     ├── PPO
|     ├── REINFORCE
|     ├── SAC
|     └── TD3
|
└── src
     ├── environment
     ├── evaluation
     ├── functions
     └── training
```

* A pasta [`imgs/`](imgs/) contém as imagens utilizadas nesse documento;
* A pasta [`policies/`](policies/) contém as políticas aprendidas pelos agentes, que representam os algoritmos de otimização aprendidos;
* A pasta [`src/`](src/) contém os códigos utilizados para o treinamento dos agentes ([`src/training`](src/training)), avaliação dos agentes ([`src/evaluation`](src/evaluation)), funções de benchmark ([`src/functions`](src/functions)) e o ambiente ([`src/environment`](src/environment)). 

### Instalação
A implementação dos códigos é feita em [`Python 3.8`](https://docs.python.org/3.8/) com o [`TensorFlow 2.5.0`](https://github.com/tensorflow/tensorflow/tree/r2.5) e [`TF-Agents 0.8`](https://github.com/tensorflow/agents/tree/r0.8.0). Para realizar executar os códigos, faz-se necessário clonar o repositório e instalar as dependências necessárias.

Primeiro, realize o clone do repositório
```shell
$ git clone https://github.com/rl-opt/rlopt
$ cd rlopt
```

Então, instale as dependências usando o `pip` (é recomendado utilizar um ambiente virtual do python)
```shell
$ pip install -r requirements.txt
```

<a id='BenchmarkFunctions'></a>
## Funções de Benchmark
Os problemas de otimização escolhidos consistem em minimizar 8 funções matemáticas (F<sub>1</sub>—F<sub>8</sub>):

<img src="/imgs/function_summary.png" alt="Descrição das Funções" width="480"/>

Essas funções são conhecidas pela literatura ([[Laguna and Martí 2005]](https://link.springer.com/article/10.1007/s10898-004-1936-z)[[Molga and Smutnicki 2005]](https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf)), sendo parte, inclusive, da [_IEEE WCCI2020 Competition on Evolutionary Multi-task Optimization_](http://www.bdsc.site/websites/MTO_competition_2020/MTO_Competition_WCCI_2020.html).

A implementação das funções se encontra em [`src/functions/`](src/functions/numpy_functions.py), os agentes (representados por suas _policies_) em [`policies`](policies/) e os códigos utilizados para o treinamento em [`src/training`](src/training). 

<a id='ConvergencePlots'></a>
## Gráficos de Convergência
Abaixo, estão os gráficos de convergência dos algoritmos de otimização aprendidos para as funções consideradas (**d=30**). Os algoritmos aprendidos (_policies_) se encontram em [`policies/`](policies/). Os resultados são apresentados em termos da média de 100 execuções distintas. Utilizamos os algoritmos Gradient Descent (GD) e Nesterov’s Accelerated Gradient (NAG) como baselines.

F<sub>1</sub>:

<img src="/imgs/convergence/F1_30D_plot.png" alt="Convergência F1" width="480"/>

F<sub>2</sub>:

<img src="/imgs/convergence/F2_30D_plot.png" alt="Convergência F2" width="480"/>

F<sub>3</sub>:

<img src="/imgs/convergence/F3_30D_plot.png" alt="Convergência F3" width="480"/>

F<sub>4</sub>:

<img src="/imgs/convergence/F4_30D_plot.png" alt="Convergência F4" width="480"/>

F<sub>5</sub>:

<img src="/imgs/convergence/F5_30D_plot.png" alt="Convergência F5" width="480"/>

F<sub>6</sub>:

<img src="/imgs/convergence/F6_30D_plot.png" alt="Convergência F6" width="480"/>

F<sub>7</sub>:

<img src="/imgs/convergence/F7_30D_plot.png" alt="Convergência F7" width="480"/>

F<sub>8</sub>:

<img src="/imgs/convergence/F8_30D_plot.png" alt="Convergência F8" width="480"/>

<a id='Comparisons'></a>
## Comparação em diferentes dimensões
Abaixo, estão as tabelas comparando os diferentes algoritmos de otimização aprendidos para as funções com diferentes dimensões (**d=5**, **d=10** e **d=30**). Os agentes foram treinados por 500 (**d=5**), 1000 (**d=10**) e 2000 (**d=30**) episódios. Os valores apresentados representam a média da solução final obtida pelos agentes em 100 execuções distintas.

O tempo médio para o treinamento dos agentes foi cerca de: 

- **2h 45min** para **2000** episódios e **d = 30**;
- **1h 45min** para **1000** episódios e **d = 10**;
- **45min** para **500** episódios e **d = 5**.

<img src="/imgs/comparisons/F1_dims.png" alt="Comparação F1" width="480"/>

<img src="/imgs/comparisons/F2_dims.png" alt="Comparação F2" width="480"/>

<img src="/imgs/comparisons/F3_dims.png" alt="Comparação F3" width="480"/>

<img src="/imgs/comparisons/F4_dims.png" alt="Comparação F4" width="480"/>

<img src="/imgs/comparisons/F5_dims.png" alt="Comparação F5" width="480"/>

<img src="/imgs/comparisons/F6_dims.png" alt="Comparação F6" width="480"/>

<img src="/imgs/comparisons/F7_dims.png" alt="Comparação F7" width="480"/>

<img src="/imgs/comparisons/F8_dims.png" alt="Comparação F8" width="480"/>
