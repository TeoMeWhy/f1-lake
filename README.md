# F1 Lake

Vamos coletar, armazenar, processar dados de Fórmula 1 para construção de análises e modelos preditivos.

Todo projeto será realizado ao vivo na [Twitch (Téo Me Why)](twitch.tv/teomewhy) de maneira aberta para toda a comunidade com início no dia 02/03.

A gravação ficará disponível no [YouTube](https://youtube.com/@teomewhy) para apoiadores do canal.

## Etapas do projeto
- [Coleta](#coleta)
- [Envio dos Dados](#envio-dos-dados)
- [Camada Bronze](#camada-bronze)
- [Camada Silver](#camada-silver)
- [Camada Gold](#camada-gold)

### Coleta

Utilizaremos a biblioteca [FastF1](https://docs.fastf1.dev/) como fonte de dados, a partir de scripts em Python para realizar a coleta das informações históricas.

Esta etapa será executada em um servidor próprio de maneira agendada.

### Envio dos dados

Ainda que a coleta seja feita em um servidor próprio, enviaremos esses dados para um Bucket S3 na AWS. Assim, a Nekt poderá acessar os dados brutos para realizar a ingestão em nosso Lakehouse.

Em termos de camada de dados, ela nos servirá de camada `raw`, ou camada de dados brutos.

### Camada Bronze

Na camada bronze, nossos dados estarão consolidados em formato `Delta` com histórico de modificações, facilitando suas consultas. Além disso, teremos uma representação fiel de como este dado poderia ser encontrado em sua origem.

### Camada Silver

A partir dos dados na camada anterior, já dentro de nosso Lakehouse, podemos realizar novas modelagens de dados e também criação de Feature Stores com o histórico de cada entidade de nosso interesse.

### Camada Gold

Aqui, deixamos apenas tabelas em formatos de relatórios e dados sumarizados para que sejam facilmente analisados e conectados em ferramentas de BI/dashboards.

### Treinamento do Modelo

Utilizando dados das Feature Store e eventos de interesse, podemos gerar uma Analytical Base Table (ABT) para treinar nossos algoritmos de Machine Learning.

Os modelos serão treinados e comparados localmente, fazendo uso do MLFlow hospedado em nosso servidor próprio.

### Aplicação para usuário

Com nosso modelo treinado, podemos criar uma aplicação onde entusiastas de Fórmula 1 poderão acompanhar as predições do modelo.
