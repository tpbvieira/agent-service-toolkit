# Agente de Otimização de Código

Este projeto implementa um agente que fornece sugestões de otimização de código com base em boas práticas de programação Python. Este agente atua exclusivamente como revisor de código Python, permitindo interações incrementais, em um processo evolutivo de melhoria de código. Para isto, o agente mantém o diálogo da thread corrente em sua memória.

O agente dispõe de mecanismos de segurança para evitar a geração de respostas indesejadas. Este recurso é opcional e está habilitado apenas se a variável GROQ_API_KEY estiver preenchida.

O agente está disponível através de uma aplicação web, para a fácil interação e melhor usabilidade através de um chat.

O projeto é baseado nas seguintes tecnologias:
 - Streamlit.
 - FastAPI
 - LangGraph
 - [Agent Service Tookit](https://github.com/JoshuaC215/agent-service-toolkit) 
 - Crewai
 - PostgreSQL
 
## Passos para execução e testes

### Agentes e assistente

Recomenda-se a utilização do docker para a execução do assistente, embora a execução através do Python também seja simples.

O desenvolvimento e testes foi feito utiizando uma GOOGLE_API_KEY que pode ser obtida gratuitamente junto ao respectivo fornecedor. 

Uma vez obtida a chave, deve ser executado o comando abaixo, substituindo your_api_key pela respectiva chave obtida.

```sh
echo 'GOOGLE_API_KEY=your_api_key' >> .env
docker compose watch
```

Como resultado teremos três containers em execução, estes são: 'code_reviewer-streamlit_app-1', 'code_reviewer-pgvector-1' e 'code_reviewer-agent_service-1'.

O assistente estará disponível em http://localhost:8501/, através de um chat que tem como agente default o code-reviwer, que é o principal objeto deste projeto.

Os tests podem ser feitos por meio da interação com o agente através do chat, ou através de chamadas à API, na forma abaixo para o método analyze-code:

   ```sh
   curl -X POST http://localhost:80/analyze-code -H "Content-Type: application/json" --insecure --data "{\"message\": \"give me a backtrack example" &
   ```

O método health pode ser testado da seguinte forma:

   ```sh
   curl -X GET http://localhost:80/health -H "Content-Type: application/json" --insecure
   ```

Os dados persistidos podem ser verificados através do acesso direto ao PostgreSQL, na porta 5432, utilizando os dados abaixo:
```
AGENT_PGVECTOR_USER = "agent_db_user"
AGENT_PGVECTOR_PWD = "4g3ntdbus3r"
AGENT_PGVECTOR_HOST = "pgvector"
AGENT_PGVECTOR_DB = "agent_db"
```

Estas credenciais estão presentes no código por simplificade no desenvolvimento e testes, mas em uma próxima iteração devem ser levadas para serem providas por variáveis de ambiente.

### Testes com Crewai

A orquestração de agentes via Crewai foi implementada deforma separada dos agentes, através de um fluxo que permite a definição de um fluxo de execução de forma desacoplada e flexível. 

Os testes requerem uma instalação do Python. Não foi verificada a compatibilidade com versõesdo Python diferentes de 3.12.8.

Para testar a orquestração de agentes via Crewai, os seguintes comandos devem ser executados:

   ```sh
   # uv is recommended but "pip install ." also works
   pip install uv
   # "uv sync" creates .venv automatically
   uv sync --frozen
   # se linux, use o comando abaixo para ativar o venv
   source .venv/bin/activate
   # se windows usar o comando abaixo no cmd para ativar o venv. NÃO usar powershell, 
   .venv\Scripts\activate
   uv pip install crewai==0.67.0 crewai-tools==0.12.1
   python src/run_code_review_flow.py

   ```

Após algum tempo, deverão ser impressas no console mensagens iniciadas com os padrões '#> chatbot_message', '#> code_reviewer_data', '#> code_reviewer_response' e '#> Reviewed code'.

Estas mensagens demonstram um diálogo entre dois agentes, chamados chatbot e code-reviewer, onde o agente generalista chatbot gera um código que faz uma implemenação simples de um merge sort, e o envia para revisão do code-reviwer, que por fim fornece sua análise e sugestões de melhorias no código recebido.

### Decisões Arquiteturais

O Agent Service Toolkit foi escolhido como template para o desenvolvimento de agentes, por já fornecer uma infraestrutura de boa qualidade arquitetural, permitindo reúso de código frequentemente replicado no desenvolvimento e testes de agentes, como também permitindo ter um foco maior nas particularidades de cada agente e nos workflows que se deseja desenvolver.

Langgraph tem sido uma das principais soluções para o desenvolvimento de Agentes baseados em LLM, atuando como uma solução que preenche lacunas e dificuldades comumente verificadas no desenvolvimento de atentes utilizando langchain, inclusive incluindo capacidades de lidar com agentes na forma de grafos, não se limitando a pipelines lineares.

Os endpoints foram implementados para respoder requisções de forma assíncrona, permitindo I/O não bloqueante, redução no número e tempo de conexões ativas,  redução na alocação de memória, maior capacidade de lidar com conexões concorrentes e maior escalabilidade. 

O assistente entrega suas respostas na forma de streaming, por default, permitindo uma melhor experiência do usuário, tem uma melhor sensação de evolução da geração de respostas, como também reduz a alocação de memória com buffer de respostas, que é principalmente nos casos de geração de textos longos.

A adoção de guardrail é um importante mecanismo de segurança e conformidade, através dele é possível ter uma verfificação independente, por modelo especialista, quanto a conforidade com requisitos de segurança para as mensagens geradas. Neste passo do worfflow, é possível incluir verificações de privacidde e outroas verificações de restrições importantes para uma organização.

A orquestração de agentes via Crewai foi implementada por meio de uma integração de baixo acomplamento, por meio de chamadas de API dos agentes, permitindo assim ter infraestruturas separadas para agentes e orquestradores, como também permitindo ter mudanças no orquestrador de agentes (Crewai) ou nos agentes, de forma independente e desacoplada. A implementação foi feita através de flows do Crewai.

O desenvolvimento baseado em containers Docker foi escolhido devido a sua facilidade de desenvolvimento e implantação em diversos ambientes, abstraindo complexidades relevantes de infraestrutura.

O desenvolvimento de testes unitários e de integração são importantes como estilo de desenvolvimento baseado em testes, com melhor qualidade e controle através de testes de regressão.

A organização de código seguiu boas práticas de modularização, em conformidade com o cookiecutter.

### Testes

Testes unitários e de integração podem ser feitos através do comando:

```sh
pytest
```

## Escalabilidade

É necessário implementar um pool de conexões para acesso a dados, de forma a suportar um maior acesso concorrente ao banco de dados e melhorar a gestão de recursos computacionais.  

Uma solução de filas e mensageria, tal como Redis, é importante para escalar e estar apto a lidar com crescimento na quantidade de requisições concorrentes. Desta forma, seria possível ter uma solução distribuída, com workers consumindo filas e fornecendo recursos necessários para a escalabilidade horizontal da solução. 

Uma solução de cache, tal como Redis, é importante para fornecer acesso à memória de baixa latência para as informações mais demandadas, tais como as memórias e estados de conversações ativas,

## Licença

MIT