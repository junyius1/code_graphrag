# Code GraphRAG

**Deterministic code knowledge graphs + Microsoft GraphRAG semantic search for source repositories.**

Code GraphRAG parses a source directory with **Tree-sitter (CST) + language ASTs**, builds a
*deterministic, source-located* Code Knowledge Graph (entities + typed relations, no LLM
involved), then feeds it into **Microsoft GraphRAG v3's indexing pipeline** through a custom
workflow. The result is a hybrid index you can query two ways:

1. **Deterministic graph queries** — precise, reproducible, no LLM: *where is X*, callers,
   callees, imports, inheritance, call paths, config impact, related files.
2. **GraphRAG semantic search** — local / global / drift / basic search over communities and
   reports for open-ended questions ("how does user registration work?").

Everything runs through **LangChain** (LLM/embedding factories), **LangGraph** (pipeline
orchestration), and optionally **Deep Agents** (an interactive code-exploration agent). A
fully **offline mock mode** (canned LLM + fixed-dimension embeddings) makes the entire
pipeline — including tests — runnable with no network or API keys.

---

## 1. Installation

Requires **Python 3.10+** (developed on 3.12) and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this-repo> code_graphrag && cd code_graphrag
uv sync --all-groups          # creates .venv, installs runtime + dev (pytest/ruff) deps
```

Verify:

```bash
.venv/bin/python -c "import code_graphrag; print(code_graphrag.__name__)"
.venv/bin/code-graphrag --help
```

The `code-graphrag` console script is the CLI entry point (Typer).

## 2. Specifying a Source

Point the tool at any directory containing source code. Discovery is
globs-based and layered:

| Layer   | Default globs                                                            | Kind     |
|---------|--------------------------------------------------------------------------|----------|
| source  | `*.py *.js *.jsx *.ts *.tsx *.mjs *.cjs *.java *.go *.rs *.c *.cc *.cpp *.h *.hpp *.cs …` | `source` |
| docs    | `*.md *.mdx *.rst *.txt`                                                 | `doc`    |
| config  | `*.json *.yaml *.yml *.toml *.ini *.cfg`                                 | `config` |

Excluded by default: `.git`, `.venv`, `venv`, `node_modules`, `__pycache__`, `build`, `dist`,
cache dirs. Files are also skipped (and recorded) when binary, unreadable, or larger than
`max_file_bytes` (default 1 MB). Every discovered file is classified `source` / `doc` /
`config` and carries a repo-relative path — that path is the anchor for all later
source locations.

```bash
code-graphrag index --source /path/to/repo --output /path/to/index
```

## 3. Building the Index

```bash
# Real LLM (OpenAI-compatible). Requires an API key:
export CODE_GRAPHRAG_LLM_API_KEY=sk-...
code-graphrag index --source ./myrepo --output ./idx \
  --llm-provider openai_compatible --llm-model gpt-4.1-mini

# Fully offline (mock LLM + mock embeddings):
code-graphrag index --source ./myrepo --output ./idx --mock
```

Key options (see `code-graphrag index --help` for the full list):

- `--llm-provider {openai|openai_compatible|anthropic|ollama|mock}`
- `--llm-model`, `--llm-api-key` / `CODE_GRAPHRAG_LLM_API_KEY`, `--llm-base-url`
- `--emb-model`, `--emb-api-key`, `--emb-base-url`, `--emb-dimensions`
- `--enrich` — optional LLM semantic enrichment of entity descriptions (off by default)
- `--max-text-unit-chars N` — sub-chunk long semantic text units
- `--max-cluster-size N` — Leiden community size
- `--exclude SEG` (repeatable) — extra path segments to skip

The pipeline runs as a 7-node **LangGraph** workflow (see §10). Output layout:

```
<output>/
  output/            # GraphRAG parquet tables + graph
    documents.parquet  text_units.parquet  entities.parquet  relationships.parquet
    communities.parquet  community_reports.parquet  graph.graphml  stats.json  context.json
  lancedb/           # vector store: entity_description, community_full_content, text_unit_text
  code_graphrag_index.json   # deterministic sidecar (entities, relations, locations, title map)
  graphrag_settings.json     # redacted GraphRagConfig for rebuilding the query side
  logs/indexing-engine.log
  cache/  input/
```

## 4. Querying

### Deterministic graph queries (no LLM)

```bash
code-graphrag query -i ./idx -q "Where is the handle method implemented?"
code-graphrag query -i ./idx -q "What are the callers of send_email?"
code-graphrag query -i ./idx -q "What does handle_register call?"
code-graphrag query -i ./idx -q "Who imports the services module?"
code-graphrag query -i ./idx -q "What class inherits from User?"
code-graphrag query -i ./idx -q "Show the call path from main to send_email"
code-graphrag query -i ./idx -q "What code is affected by the storage_path config?"
code-graphrag query -i ./idx -q "Which files are involved in UserService?"
```

The router maps question phrasing to a query mode automatically; force one with `--mode
{where|callers|callees|imports|imported_by|inherits|config|files|path}`. Add `--json` for
machine-readable output (`question`, `source`, `answer`, `entities[]` with `file:line`,
`path[]`).

### GraphRAG semantic search (LLM)

```bash
code-graphrag query -i ./idx -q "How does user registration work end to end?" --mode local
code-graphrag query -i ./idx -q "What is this repository about?" --mode global
# drift / basic also supported; --community-level N
```

Mock-mode indexes answer offline with canned responses (useful for plumbing; a real model
answers meaningfully).

### Inspect & explore

```bash
code-graphrag inspect -i ./idx --entity UserService        # entity + its relations

# Deep Agents interactive exploration (see §11):
code-graphrag explore -i ./idx -s ./myrepo -q "Walk me through the auth flow" \
  --llm-provider openai_compatible --llm-model gpt-4.1-mini
```

## 5. Architecture

```
 source dir
     │  discover (globs, filters)                     [discovery/]
     ▼
 DiscoveredFile[]  ──  parse (tree-sitter CST + lang AST)  [parsing/]
     │  RawStructure (defs/imports/calls/raises/spans/decorators)
     ▼
 Code Semantic Model  ──  build (two-pass: symbol index → entities/relations)  [semantic/]
     │  CodeKnowledgeGraph: stable-id entities + typed relations (file:line)
     ▼
 GraphRAG tables (documents/text_units/entities/relationships)  [graphrag/tables.py]
     │  custom workflow `load_code_graph`
     ▼
 GraphRAG v3 pipeline: create_final_documents → finalize_graph → create_communities
     → create_final_text_units → create_community_reports → generate_text_embeddings
     │  (Microsoft GraphRAG, LLM for reports+embeddings)
     ▼
 index (parquet + lancedb + graphml + sidecar + redacted settings)
     │
     ├── CodeGraphQuery   (networkx over sidecar)  → deterministic answers
     ├── graphrag_search  (local/global/drift/basic) → semantic answers
     └── Deep Agents agent (tools over both + filesystem)  → open-ended exploration
```

Design principles:

- **Determinism first.** The knowledge graph is built purely by static analysis: same input
  → same entities/relations/ids. LLMs only *add* (community reports, embeddings, optional
  enrichment); they never create the core graph.
- **Stable identity.** Entity ids are `sha1(type|repo|file|qualified_name)[:20]`, so graphs
  are diffable across runs and merges don't churn.
- **GraphRAG as a consumer, not an authority.** We *replace* GraphRAG's LLM entity
  extraction with our deterministic tables; we keep its (excellent) community detection,
  reporting, embedding, and query engines.
- **Everything cites `file:line`.** The sidecar maps GraphRAG titles back to entities with
  exact spans.

## 6. From CST to the Semantic Model

Parsing fuses two passes per source file (`parsing/extractors/`):

1. **Tree-sitter CST** (`parsing/languages.py` holds a `ParserPool` of 9 grammars) gives
   precise *structure and spans*: definitions (function/class/method/interface/…), their
   `Span` (1-based line ranges), decorators, base classes, and parameters (including
   annotations, defaults, `*args`/`**kwargs`).
2. **Language AST** (Python uses stdlib `ast`; other languages use tree-sitter node tables)
   gives *behavioral facts* that are awkward in the CST: `import` statements, call sites,
   `raise` targets, module-level constant assignments, and docstrings.

Both are merged into a `RawStructure` (`parsing/common.py`):

```
RawStructure
 ├─ definitions:  RawDefinition  (kind, name, qualified_name, span, params, decorators, base, docstring, is_test/cli/api)
 ├─ imports:      RawImport      (module, level, symbols, alias, line)
 ├─ calls:        RawCall        (callee, is_method_call, receiver, args, line)
 ├─ raises:       RawRaise       (exception, line)
 └─ assignments:  RawAssignment  (name, value_preview, is_class_attr, is_constant)
```

The semantic builder (`semantic/builder.py`) then runs **two passes**:

1. **Index pass** — collect every definition across all files into a `SymbolIndex`
   (module→file, qualified name→file+span, class→methods, file→imports) so that calls,
   imports, and inheritance can be *resolved across files* before anything is emitted.
2. **Emit pass** — walk each file and emit entities + relations, resolving receivers
   (`self.users.register(...)` → `UserService.register`), import aliases, and relative
   imports. Unresolved external names (e.g. `ValueError`) become `__external__` exception
   entities rather than being dropped.

## 7. Building the Code Graph

Entities and relations use controlled vocabularies (`config/models.py`,
`GraphRAGAdapterConfig`):

- **Entity types:** `repository, directory, file, module, package, class, function, method,
  parameter, variable, constant, api, cli, command, exception, test, documentation,
  configuration, concept`
- **Relation types:** `CONTAINS, DEFINES, IMPORTS, CALLS, INHERITS, IMPLEMENTS, USES,
  RETURNS, ACCEPTS_PARAMETER, RAISES, TESTS, DOCUMENTED_BY, CONFIGURED_BY, REFERENCES,
  RELATED_TO`

Every entity carries a stable id, `name`, `qualified_name`, `file`, and 1-based
`start_line`/`end_line`; every relation carries source/target ids, a type, a human
description (e.g. `CALLS: main calls NotificationService.send_email`), and a weight.
Relations are deduplicated by `(source, target, type)` with summed weight.

Example (sample project): ~90 entities, ~190 relations —
`UserService.register CALLS App.handle_register`, `User INHERITS BaseUser`,
`test_register TESTS UserService.register`, etc.

## 8. GraphRAG Integration

GraphRAG **v3** changed its API substantially (no more `create_graphrag_storage`,
`create_base_model_function`, or `load_default_config`). We adapt with:

- **One custom workflow**, `load_code_graph`, registered on `PipelineFactory`. It reads
  `context.state["additional_context"]["code_graphrag_tables"]` (the DataFrames we built)
  and writes the `documents`, `text_units`, `entities`, and `relationships` tables to
  `context.output_table_provider`.
- **A custom pipeline** set via `config.workflows` that *skips* GraphRAG's LLM extraction
  (`load_input_documents`, `create_base_text_units`, `extract_graph`, `extract_covariates`)
  and keeps its deterministic/LLM stages:
  `load_code_graph → create_final_documents → finalize_graph → create_communities →
  create_final_text_units → create_community_reports → generate_text_embeddings`.
- **Two compatibility shims** we discovered empirically:
  - `cluster_graph.use_lcc` is forced to **False** — GraphRAG's `stable_lcc` *uppercases*
    node names, which breaks the case-sensitive title→entity merge for code
    (`App.handle` vs `APP.HANDLE`).
  - Every relationship row is guaranteed **≥1 `text_unit_ids`** — empty lists become NaN
    after `explode()` and crash `sorted(set(...))` in `create_communities`.
- **Mock mode** (`--mock`) maps to GraphRAG's built-in mock completion + mock embedding
  (16-dim) so the whole pipeline runs offline. The saved `graphrag_settings.json` redacts
  `api_key` but **preserves** `mock_responses` so queries can rebuild the config later.

Text units are created at **semantic boundaries** (one per function/class/constant, plus a
repo-level unit), sub-chunked at `max_text_unit_chars`.

## 9. Using LangChain

LangChain (v1) is the provider-agnostic model layer. Factories live in `llm/models.py`:

```python
from code_graphrag.llm.models import create_chat_model, create_embeddings
from code_graphrag.config.models import LLMConfig, EmbeddingConfig, LLMProvider

chat = create_chat_model(LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4.1-mini"))
# -> langchain.chat_models.init_chat_model("openai:gpt-4.1-mini", temperature=0, ...)

emb = create_embeddings(EmbeddingConfig(provider="openai", model="text-embedding-3-small"))
```

Structured output (used by optional enrichment, `llm/enrichment.py`) uses
`model.with_structured_output(SymbolSummary)` and invokes with
`[SystemMessage(...), HumanMessage(...)]`. `provider=mock` returns a deterministic
`BaseChatModel` (`_MockChatModel`) that also supports a minimal tool-calling loop so the
Deep Agents path is testable offline. **You** are free to build your own LangChain chains
on top of `create_chat_model`; nothing in the core is coupled to a specific partner.

## 10. Using LangGraph

Both pipelines are **LangGraph `StateGraph`s** (nodes are `async fn(state) -> dict`):

- **Index** (`workflow/index_workflow.py`) — 7 nodes:
  `discover_files → parse_source → build_semantic_model → extract_semantics →
  build_graph_data → build_graphrag_index → finalize`, wired `START → … → END` and invoked
  via `await graph.ainvoke(input, {"configurable": {"thread_id": ...}})`.
  Public entry: `run_index_workflow(source, output, config)`.
- **Query** (`workflow/query_workflow.py`) — 3 nodes:
  `classify → answer → format`. `classify` routes the question (deterministic mode vs
  GraphRAG search), `answer` dispatches to `CodeGraphQuery` or `agraphrag_search` (the
  **async** wrapper — important, since nodes run inside a live event loop), and `format`
  appends the source. Public entry: `run_query_workflow(index_dir, question, forced_mode)`.

State is a `TypedDict`; lists use `Annotated[list, operator.add]` reducers where
accumulation is needed.

```python
import asyncio
from code_graphrag.workflow.query_workflow import run_query_workflow

state = asyncio.run(run_query_workflow("./idx", "Where is handle implemented?"))
print(state["route"], state["answer"])
```

## 11. Using Deep Agents

An **optional** interactive layer (`agents/code_agent.py`) builds a
`create_deep_agent` with:

- the built-in filesystem tools (`ls`, `read_file`, `glob`, `grep`, …) rooted at the
  **source repo** (`FilesystemBackend(root_dir=..., virtual_mode=True)`) so the agent can
  read real code;
- **custom code-graph tools** (`where_is`, `callers`, `callees`, `imports_of`,
  `imported_by`, `inheritance`, `call_path`, `related_files`, `config_impact`,
  `code_overview`) plus `graphrag_local_search` / `graphrag_global_search`;
- an `architecture-analyst` subagent for high-level structure questions.

```python
from code_graphrag.agents import run_code_agent, create_code_agent
from code_graphrag.config.models import LLMConfig, LLMProvider

ans = run_code_agent("./idx", "Walk me through the auth flow",
                     source_dir="./myrepo", llm_config=LLMConfig(provider=LLMProvider.OPENAI))
```

Or drive it yourself: `agent = create_code_agent(...)` then
`agent.invoke({"messages": [{"role": "user", "content": "..."}]})` and read the last
`AI` message. The CLI exposes it as `code-graphrag explore`.

## 12. Adding a New Language

Languages plug in at two seams:

1. **Grammar + spec** — add a `Language` member (if new) and a `LangSpec` in
   `parsing/languages.py` (`LANG_SPECS`), mapping source globs → `LangSpec(language,
   node tables)`. `language_for_path()` resolves a file to its spec; the `ParserPool`
   lazily loads the tree-sitter grammar.
2. **Extractor** — implement `extract_<lang>(source, rel_path) -> RawStructure`
   (see `parsing/extractors/generic.py` for the per-language node tables for
   JS/TS/Java/Go/Rust/C/C#/C++). Register it in `parsing/parse.py`'s dispatch (the
   `if spec.language is Language.PYTHON` branch).

Because the rest of the system consumes only `RawStructure`, a new language gets the full
semantic model, GraphRAG tables, communities, and both query paths for free. Keep the
extractor defensive: return a `RawStructure` with `parse_error=True` on failure rather
than raising (a single bad file must never abort a run).

## 13. Adding New Entity / Relation Types

1. **Vocabulary** — add the name to `GraphRAGAdapterConfig.entity_types` and/or
   `relationship_types` in `config/models.py`. (These are validated vocabularies, not
   hard constraints, so queries still work if you emit an off-list type.)
2. **Emit** — in `semantic/builder.py`, call
   `self.graph.entity("<type>", name, qualified_name, file, ...)` and
   `self.graph.relate(src, tgt, "<REL>", description, weight)`. Resolution uses the
   `SymbolIndex`; use `__external__` as the file for unknown external symbols.
3. **Query** — to expose the new relation to deterministic queries, add a handler in
   `query/queries.py` (mirror `callers`/`callees` using `_neighbors(id, TYPE, "out"/"in")`)
   and a routing keyword in `query/routing.py` (`detect_deterministic_mode`).
4. **Rebuild** — entity ids derive from `(type, repo, file, qualified_name)`, so changing
   the type of an existing entity changes its id; rebuild the index after type changes.

## Tests

The suite (59 tests, `pytest`) runs **fully offline** against `tests/sample_project`:

```bash
.venv/bin/python -m pytest
```

Coverage spans: file discovery & filtering, Python parsing (tree-sitter + AST: functions,
classes, methods, imports, calls, raises, constants, parameters/defaults, control-flow
nested calls, source spans, bad-code tolerance), stable entity ids, entity/relation
deduplication, full graph construction (with graph integrity + inheritance chain),
GraphRAG table structure + config (mock pipeline, `use_lcc=False`, redaction), end-to-end
mock indexing, deterministic queries + routing, GraphRAG search (local/global/missing),
LangGraph index & query workflows, the CLI (index/query/inspect), and the Deep Agents
integration (tools + mock tool-calling model).

## Project layout

```
src/code_graphrag/
  config/       # pydantic config models (IndexConfig, LLMConfig, …)
  discovery/    # recursive globs + filtering -> DiscoveredFile[]
  parsing/      # languages.py (grammar pool), common.py (RawStructure), extractors/
  semantic/     # model.py (stable ids, graph), builder.py (two-pass emission)
  graphrag/     # tables.py (-> DataFrames), adapter.py (GraphRAG v3 pipeline)
  query/        # queries.py (deterministic), routing.py, graphrag_search.py
  workflow/     # index_workflow.py, query_workflow.py (LangGraph)
  llm/          # models.py (LangChain factories), enrichment.py (optional)
  agents/       # code_agent.py (Deep Agents)
  cli/          # app.py (Typer: index / query / inspect / explore)
```

## References

- Microsoft GraphRAG v3 (`graphrag`), LangChain v1, LangGraph, Deep Agents, Tree-sitter
  + `tree-sitter-language-pack`, NetworkX, PyArrow/LanceDB, Typer.
