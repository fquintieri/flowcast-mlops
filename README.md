# Flowcast MLOps: Previsão de Demanda Varejista

## Objetivo do Projeto

O Flowcast é uma plataforma de MLOps desenvolvida para automatizar de ponta a ponta o ciclo de vida de modelos de Machine Learning aplicados à previsão de demanda (Demand Forecasting). O objetivo do negócio é fornecer predições precisas de vendas futuras, mitigando problemas de ruptura de estoque (out-of-stock) e excesso de inventário (overstock).

Sob a ótica técnica, o projeto demonstra a implementação de uma arquitetura escalável e reprodutível que elimina a necessidade de intervenção manual em operações de rotina. A solução integra a coleta de dados, processamento de variáveis temporais (feature engineering com DuckDB), rastreamento de experimentos e governança de modelos (MLflow), orquestração de pipelines (Apache Airflow) e visualização dos resultados para o usuário final (Streamlit). O sistema é projetado para garantir que o algoritmo com a melhor performance comprovada esteja sempre operando em produção.

---

## Arquitetura, Funcionamento e Tempestividade (DAGs)

A orquestração do sistema é gerenciada pelo Apache Airflow, que coordena os containers Docker por meio de dois pipelines independentes com ciclos de execução distintos (tempestividade):

> **Sobre os dados:** o ambiente **simula** o ciclo de MLOps sobre um dataset **estático** do Kaggle (competição [`demand-forecasting-kernels-only`](https://www.kaggle.com/competitions/demand-forecasting-kernels-only)). Não há ingestão incremental: a cada execução, o `data_prep.py` faz **carga completa** (`CREATE OR REPLACE`) das tabelas `raw_sales` e `raw_test` no DuckDB a partir dos CSVs. O download só acontece se `data/raw/train.csv` ainda não existir. As features também são recalculadas por inteiro. Como os dados não mudam entre execuções, retreinos sucessivos tendem a produzir modelos equivalentes ao Champion, e a regra de promoção vai rejeitá-los.

### 1. Rotina Diária: Inferência (`flowcast_daily_inference`)

* **Tempestividade:** Executada todos os dias às 02:00 (`0 2 * * *`).
* **Objetivo:** Garantir que as previsões estejam sempre atualizadas para a operação diária.
* **Fluxo:** Recarrega os dados brutos (carga completa) e recalcula as variáveis temporais (calendário e lags). Em seguida, carrega a versão do modelo com alias `@Champion` no Model Registry do MLflow e gera as predições para o período de teste. O arquivo `submission.csv`, consumido pelo dashboard, é substituído de forma atômica. **Sem Champion, a inferência falha explicitamente**: ela nunca usa um modelo não aprovado. O cold start é tratado apenas pela rotina semanal.

### 2. Rotina Semanal: Retreinamento Contínuo (`flowcast_weekly_retrain`)

* **Tempestividade:** Executada aos domingos às 04:00 (`0 4 * * 0`).
* **Objetivo:** Demonstrar o retreinamento periódico para combater o desgaste temporal do modelo (Data/Concept Drift). Em um cenário real, isso seria feito com o histórico recém-acumulado.
* **Fluxo:** Recria o pipeline de dados e treina um novo modelo candidato (Challenger). O Challenger e o Champion atual são avaliados **no mesmo holdout**, e o Challenger só é promovido se melhorar a métrica de decisão além da margem mínima, sem piorar o bias além do limite (veja [Decisões de arquitetura](#decisões-de-arquitetura)). A decisão é publicada em JSON e roteada por um `BranchPythonOperator`. Se rejeitado, o Champion continua em produção. No primeiro ciclo (sem Champion), o Challenger é promovido por W.O.

---

## Decisões de arquitetura

### Lags a partir de 91 dias (sem vazamento)
O período de teste tem 90 dias. Ao prever o último dia, a venda mais recente conhecida está 90 dias antes. Por isso, todas as features de lag (`lag_91`, `lag_98`, `lag_105`, `lag_112`, `lag_182`, `lag_364`, em `ml_agent/src/feature_engineering.py`) usam defasagem **≥ 91 dias**. Assim, nenhuma linha de treino ou de teste depende de uma venda que não estaria disponível no momento da previsão. Pelo mesmo motivo, não há médias móveis curtas.

### Pool de escrita do DuckDB
O DuckDB aceita apenas **um processo** com o arquivo aberto em escrita, e enquanto isso nenhum outro processo consegue abri-lo, nem em `read_only`. Além de os horários estarem separados (diário 02:00, semanal domingo 04:00), toda tarefa que abre o `database.duckdb` (ingestão, features, treino, avaliação e predição) roda no pool `duckdb_writer`, que tem **1 slot**. O pool é criado pelo `airflow-init`. Se uma execução atrasar e encostar na outra, as tarefas esperam na fila (`Not scheduling since there are 0 open slots in pool duckdb_writer`) em vez de falhar por lock.

### Avaliação no mesmo holdout
A comparação entre Challenger e Champion é refeita **no momento da decisão**: os dois modelos são carregados e avaliados sobre o mesmo conjunto de holdout. Não se usam as métricas que cada um registrou no próprio treino, porque elas vieram de janelas diferentes. O holdout são os **últimos `HOLDOUT_DAYS` dias** (padrão 90) relativos à data máxima de `features_sales`. Ele é definido em um único lugar (`ml_agent/src/config.py`) e reutilizado pelo treino (validação e early stopping) e pela avaliação. Cada modelo é avaliado com as features do próprio treino (`booster.feature_name()`).

> Limitação conhecida: o Challenger faz early stopping nessa mesma janela, o que dá a ele uma pequena vantagem sobre o Champion na comparação.

### Métrica única com margem e trava de bias
Uma única métrica, `DECISION_METRIC` (padrão `smape`; aceita `mae` e `rmse`), é usada no **early stopping** do LightGBM (como `feval` customizada) e na **decisão de promoção**. O Challenger só é promovido se:
1. `métrica_challenger <= métrica_champion × (1 − MIN_IMPROVEMENT)`, ou seja, melhora relativa mínima (padrão 1%);
2. `|bias_challenger| − |bias_champion| <= MAX_BIAS_DEGRADATION`, ou seja, o bias absoluto não piora além do limite (padrão 0,5 unidade).

A métrica, a margem, o limite, os scores dos dois modelos no holdout e o resultado (`decision`, `decision_reason`) ficam registrados no run do Challenger no MLflow.

### Rastreio por `run_id`
Nenhuma etapa busca "a execução mais recente". O `train.py` publica `{"run_id": "..."}` como última linha do stdout, e o Airflow repassa esse valor via XCom. O `run_id` é validado como 32 hexadecimais antes de entrar no comando. O `evaluate_model.py --run-id` avalia exatamente esse run e publica a decisão estruturada:

```json
{"decision": "promote", "reason": "improvement", "run_id": "...", "metric": "smape", "challenger": {...}, "champion": {...}, ...}
```

O `BranchPythonOperator` lê esse JSON e falha explicitamente se a saída for inválida ou se o `run_id` avaliado diferir do treinado. O `promote_model.py --run-id` só promove runs marcados com `decision=promote`. Um treino manual entre as tarefas, portanto, não altera o modelo promovido.

### Reprodutibilidade
Cada run de treino recebe as tags `git_commit`, `data_rows`, `data_train_rows`, `data_holdout_rows`, `data_max_date` e `data_sha256` (hash do conteúdo de `features_sales`). O hash do commit é lido do `.git`, montado em modo somente leitura no container de treino. Esse método não detecta alterações locais não commitadas.

---

## Variáveis de ambiente

Configure no arquivo `.env` (modelo em `.env.example`):

| Variável | Obrigatória | Padrão | Descrição |
| --- | --- | --- | --- |
| `AIRFLOW_SECRET_KEY` | Sim | — | Chave de sessão do webserver e do scheduler do Airflow. O `docker compose` falha se estiver vazia. Gere com `python -c "import secrets; print(secrets.token_hex(32))"`. |
| `KAGGLE_API_TOKEN` | Para o 1º download | — | Token da API do Kaggle. |
| `MLFLOW_TRACKING_URI` | Não | — | URI do MLflow para o serviço utilitário `ml_agent`. As DAGs usam `http://flowcast_mlflow_ui:5000`. |
| `MODEL_NAME` | Não | `flowcast_demand_model` | Nome do modelo no Model Registry, usado pelo serviço `ml_agent`. |
| `HOLDOUT_DAYS` | Não | `90` | Tamanho da janela de holdout, relativa à data máxima dos dados. |
| `DECISION_METRIC` | Não | `smape` | Métrica única de early stopping e decisão: `smape`, `mae` ou `rmse`. |
| `MIN_IMPROVEMENT` | Não | `0.01` | Melhora relativa mínima para promover (0,01 = 1%). |
| `MAX_BIAS_DEGRADATION` | Não | `0.5` | Piora máxima tolerada em \|bias\|, em unidades de venda. |

Os padrões dos parâmetros de ML ficam apenas em `ml_agent/src/config.py`. O scheduler do Airflow repassa aos containers somente as variáveis definidas.

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
* Arquivo `.env` configurado a partir do `.env.example` com os tokens de acesso necessários, incluindo `AIRFLOW_SECRET_KEY` (veja [Variáveis de ambiente](#variáveis-de-ambiente)).

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