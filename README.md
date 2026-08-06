# Flowcast MLOps: Previsão de Demanda Varejista

## Objetivo do Projeto

O Flowcast é uma plataforma de MLOps desenvolvida para automatizar de ponta a ponta o ciclo de vida de modelos de Machine Learning aplicados à previsão de demanda (Demand Forecasting). O objetivo do negócio é fornecer predições precisas de vendas futuras, mitigando problemas de ruptura de estoque (out-of-stock) e excesso de inventário (overstock).

Sob a ótica técnica, o projeto demonstra a implementação de uma arquitetura escalável e reprodutível que elimina a necessidade de intervenção manual em operações de rotina. A solução integra a coleta de dados, processamento de variáveis temporais (feature engineering com DuckDB), rastreamento de experimentos e governança de modelos (MLflow), orquestração de pipelines (Apache Airflow) e visualização dos resultados para o usuário final (Streamlit). O sistema é projetado para garantir que o algoritmo com a melhor performance comprovada esteja sempre operando em produção.

---

## Arquitetura, Funcionamento e Tempestividade (DAGs)

A orquestração do sistema é gerenciada pelo Apache Airflow, que coordena os containers Docker por meio de dois pipelines independentes com ciclos de execução distintos (tempestividade):

### 1. Rotina Diária: Inferência (`flowcast_daily_inference`)

* **Tempestividade:** Executada todos os dias (`@daily`).
* **Objetivo:** Garantir que as previsões estejam sempre atualizadas para a operação diária.
* **Fluxo:** O pipeline realiza a ingestão incremental de novos dados brutos e recalcula variáveis temporais (lags, rolling windows). Em seguida, carrega dinamicamente a versão do modelo classificada como "Champion" no Model Registry do MLflow e gera predições atualizadas para os próximos dias, sobrescrevendo o arquivo `submission.csv` consumido pelo dashboard.

### 2. Rotina Semanal: Retreinamento Contínuo (`flowcast_weekly_retrain`)

* **Tempestividade:** Executada uma vez por semana (`@weekly`).
* **Objetivo:** Combater o desgaste temporal do modelo (Data/Concept Drift) treinando novos algoritmos com o histórico de dados recém-acumulado.
* **Fluxo:** Um pipeline computacionalmente intensivo que recria todo o pipeline de dados e treina um novo modelo candidato (Challenger). O fluxo incorpora uma etapa crítica de roteamento condicional (Branching): o Challenger é avaliado contra o Champion atual em produção. O algoritmo é promovido de forma autônoma ao status de Champion apenas se demonstrar métricas de erro (como RMSE) inferiores ao modelo em uso. Caso contrário, é rejeitado, mantendo o ambiente de produção estável.

---

## Estrutura do Repositório

A topologia do projeto foi desenhada visando a separação de responsabilidades entre orquestração, modelagem, dados e front-end:

```
├── airflow/                   # Orquestração do sistema da execução dos códigos
├── dashboard/                 # Código-fonte do painel interativo em Streamlit
├── data/                      # Persistência mapeada em volumes
├── ml_agent/                  # Módulo base com scripts Python de Machine Learning
├── .env.example               # Template de variáveis de ambiente para novos ambientes
├── .gitignore                 # Arquivos ignorados pelo controle de versão
├── docker-compose.yml         # Manifesto de infraestrutura como código
└── README.md                  # Documentação oficial do projeto
```

---

## Serviços e URLs

Com a infraestrutura em execução, a rede local expõe as seguintes portas e interfaces:

| Serviço | Descrição | URL de Acesso |
| --- | --- | --- |
| **Apache Airflow** | Orquestrador de tarefas, monitoramento de DAGs e logs. | http://localhost:8080 |
| **MLflow UI** | Interface de governança, tracking de experimentos e Model Registry. | http://localhost:5000 |
| **Streamlit Dashboard** | Aplicação web para visualização gerencial das predições geradas. | http://localhost:8501 |

*As credenciais padrão para o Apache Airflow são: Usuário `admin` | Senha `admin`.*

---

## Como Executar o Projeto

### Pré-requisitos

* Docker e Docker Compose instalados no servidor ou máquina local.
* Arquivo `.env` configurado a partir do `.env.example` com os tokens de acesso necessários.

### Inicialização (Ligar)

Para provisionar a infraestrutura e iniciar todos os microsserviços de forma desanexada do terminal, execute o comando na raiz do projeto:

```bash
docker compose up -d --build
```

*Aguarde a inicialização do banco de dados do Airflow durante a primeira execução para que a porta 8080 responda adequadamente.*

### Encerramento (Desligar)

Para interromper a execução dos containers sem perder nenhum dado salvo, modelo treinado ou histórico de orquestração (preservando o estado dos volumes):

```bash
docker compose down
```

### Limpeza Completa (Hard Reset)

Caso seja necessário reverter o ambiente ao seu estado zero, destruindo a infraestrutura e apagando todos os bancos de dados, previsões geradas e logs do Airflow simultaneamente:

```bash
docker compose down -v
```