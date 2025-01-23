-- CREATE DATABASE agent_db;

-- Switch to the target database context
\c agent_db

CREATE EXTENSION IF NOT EXISTS vector;

ALTER DATABASE agent_db SET search_path TO public;
ALTER USER agent_db_user IN DATABASE agent_db SET search_path TO public;

CREATE TABLE IF NOT EXISTS analysis_history (
    id SERIAL PRIMARY KEY,
    code_snippet TEXT,
    suggestions TEXT,
    created_at timestamptz DEFAULT now()
);
