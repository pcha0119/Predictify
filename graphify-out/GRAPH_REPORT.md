# Graph Report - .  (2026-04-10)

## Corpus Check
- 37 files · ~18,144 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 309 nodes · 436 edges · 30 communities detected
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 59 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `BaseForecaster` - 26 edges
2. `QuarantineLog` - 19 edges
3. `LassoForecaster` - 14 edges
4. `RidgeForecaster` - 14 edges
5. `ElasticNetForecaster` - 13 edges
6. `Backend services — persistence and other shared infrastructure.` - 12 edges
7. `ForecastingPipeline` - 11 edges
8. `NaiveLastValue` - 10 edges
9. `NaiveRollingMean` - 10 edges
10. `NaiveSeasonalWeekly` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Lasso and Elastic Net forecasters.  Why these after Ridge: - Lasso can zero-o` --uses--> `BaseForecaster`  [INFERRED]
  backend\models\lasso_elnet_model.py → backend\models\forecaster.py
- `Lasso regression with cross-validated alpha selection using TimeSeriesSplit.` --uses--> `BaseForecaster`  [INFERRED]
  backend\models\lasso_elnet_model.py → backend\models\forecaster.py
- `Return feature names with non-zero Lasso coefficients.` --uses--> `BaseForecaster`  [INFERRED]
  backend\models\lasso_elnet_model.py → backend\models\forecaster.py
- `Elastic Net regression with cross-validated alpha and l1_ratio selection.` --uses--> `BaseForecaster`  [INFERRED]
  backend\models\lasso_elnet_model.py → backend\models\forecaster.py
- `Ridge regression with cross-validated alpha selection.      Why Ridge for this` --uses--> `BaseForecaster`  [INFERRED]
  backend\models\ridge_model.py → backend\models\forecaster.py

## Hyperedges (group relationships)
- **Artifact Storage Pipeline** — backend_data_cleaning_cleaner_py, backend_feature_engineering_features_py, backend_models_forecaster_py, concept_csv_only_artifact [INFERRED 0.85]
- **LLM In-Process Invocation Pattern** — backend_llm_gateway_py, backend_routes_llm_py, concept_llm_inprocess [EXTRACTED 1.00]
- **No-Leakage Feature Guard** — backend_feature_engineering_features_py, concept_no_future_leakage, backend_models_forecaster_py [INFERRED 0.80]

## Communities

### Community 0 - "Model Baselines and Base Classes"
Cohesion: 0.11
Nodes (13): ABC, BaseForecaster, get_all_baselines(), NaiveLastValue, NaiveRollingMean, NaiveSeasonalWeekly, Naive baseline forecasters.  Every ML model must beat these before being consi, Forecast = last observed value, repeated for every future step.     Simple but (+5 more)

### Community 1 - "LLM Gateway and Tool Execution"
Cohesion: 0.13
Nodes (25): _audit_log(), _build_tool_declarations_gemini(), _build_tools_openai(), execute_tool(), load_config(), LLM Gateway — Multi-provider adapter with tool-calling for Predictify.  Supports, Make objects JSON-serializable., Execute a tool by name. Returns a JSON string result.     app_state and prep_sta (+17 more)

### Community 2 - "Data Cleaning Pipeline"
Cohesion: 0.11
Nodes (21): aggregate_to_daily(), build_fact_sales(), _cast_numeric(), clean_header(), clean_item_master(), clean_sale_lines(), Data cleaning pipeline.  Steps per sheet:   1. Rename columns → snake_case us, Clean the Header sheet into a slim receipt-level enrichment table.      Used o (+13 more)

### Community 3 - "Config and External Connectors"
Cohesion: 0.08
Nodes (16): Central configuration for the forecasting pipeline. All paths, thresholds, and, connect_api(), connect_database(), Routes: POST /connect/api, POST /connect/database, Connect to a database and import data.      Body:       db_type  : str (postgres, Fetch data from a REST API / OData endpoint.      Body:       url         : str, QuarantineLog — accumulates rejected rows during the cleaning pipeline.  Rathe, Ridge Regression forecaster.  Uses sklearn Pipeline(StandardScaler → RidgeCV) (+8 more)

### Community 4 - "Forecasting Pipeline"
Cohesion: 0.13
Nodes (17): _build_forecast_feature_row(), fit(), ForecastingPipeline, ForecastResult, predict(), Core abstractions for the forecasting pipeline.  BaseForecaster     — ABC that, Split sorted DataFrame into train / val / test by time fraction., Generate (train_df, val_df) pairs for expanding-window walk-forward CV.      E (+9 more)

### Community 5 - "Lasso and ElasticNet Models"
Cohesion: 0.13
Nodes (10): _choose_cv(), ElasticNetForecaster, LassoForecaster, Lasso and Elastic Net forecasters.  Why these after Ridge: - Lasso can zero-o, Return feature names with non-zero Lasso coefficients., Elastic Net regression with cross-validated alpha and l1_ratio selection., Lasso regression with cross-validated alpha selection using TimeSeriesSplit., Synchronous pipeline runner — executed in a thread pool to avoid blocking the ev (+2 more)

### Community 6 - "Evaluation Metrics"
Cohesion: 0.15
Nodes (18): bias(), compare_models(), compute_all_metrics(), mae(), mape(), Evaluation metrics for the forecasting pipeline.  Metric choices ------------, Compute all metrics and return as a flat dict ready for JSON serialisation., Produce a model comparison table from a list of metric dicts.      Rows  = mod (+10 more)

### Community 7 - "Data API Routes"
Cohesion: 0.14
Nodes (17): _column_profile(), data_prep_reset(), data_preview(), data_save(), data_summary(), data_transform(), _load_prep_data(), _push_undo() (+9 more)

### Community 8 - "Ridge Regression Model"
Cohesion: 0.15
Nodes (12): Ridge regression with cross-validated alpha selection.      Why Ridge for this, Fit on non-NaN rows only.  Stores feature names for coefficient lookup., Return feature importances as a Series indexed by feature name.         Values, RidgeForecaster, _check_data(), main(), _parse_args(), CLI entry point for the Sales Forecasting pipeline.  Usage -----   python ru (+4 more)

### Community 9 - "Data Ingestion and Loading"
Cohesion: 0.16
Nodes (13): _drop_unnamed_cols(), get_workbook_summary(), load_flat_data(), load_workbook(), Data ingestion: load, validate, and summarise the Excel workbook.  Responsibil, Return a metadata summary dict suitable for the /data/summary API endpoint., Validate that a flat DataFrame has the required columns for the pipeline., Load flat data (CSV / JSON / XML or a DataFrame) and construct synthetic     sa (+5 more)

### Community 10 - "Feature Engineering"
Cohesion: 0.19
Nodes (13): add_calendar_features(), add_indian_festival_flags(), _add_lag_rolling_for_group(), build_features_for_grain(), _clip_lags_to_series_length(), get_feature_columns(), Feature engineering pipeline.  Design rules (enforced throughout):   - ZERO f, Remove lags so large that fewer than *min_keep* rows would survive.     E.g. wi (+5 more)

### Community 11 - "Job Persistence SQLite"
Cohesion: 0.22
Nodes (12): cleanup_old_jobs(), delete_job(), init_db(), list_jobs(), load_job(), Job persistence layer using SQLite for saving/loading prepared data between sess, List all saved jobs (without loading data)., Keep only the N most recent jobs. (+4 more)

### Community 12 - "Catalog API Routes"
Cohesion: 0.25
Nodes (5): item_forecast_summary(), list_items(), Routes: /stores, /items, /items/forecast_summary, /categories, Return all items with UOM, description, category, and recent sales stats., Production planning summary: forecasted daily qty per item over the given horizo

### Community 13 - "Forecasts API Routes"
Cohesion: 0.25
Nodes (7): export_forecasts(), get_forecasts(), get_metrics(), Routes: /forecasts/{grain}, /forecasts/{grain}/export, /evaluation/metrics, Download forecast as XLSX., Return evaluation metrics, optionally filtered by model name., Return forecast + actuals for a given grain, horizon, and group (Plotly-ready).

### Community 14 - "Jobs API Routes"
Cohesion: 0.25
Nodes (5): load_saved_job(), Routes: /jobs/save, /jobs/list, /jobs/load/{job_id}, DELETE /jobs/{job_id}, Save the current prep state (DataFrame) to the database for later resumption., Load a previously saved job from the database and restore it to prep state., save_current_job()

### Community 15 - "LLM Chat API Routes"
Cohesion: 0.25
Nodes (7): chat_endpoint(), llm_config_get(), llm_config_update(), Routes: POST /chat, GET /llm/config, POST /llm/config, Stream an LLM response via Server-Sent Events.      Body:       messages      :, Return current LLM configuration (API keys redacted)., Update LLM configuration.      Body (all optional):       active_provider, activ

### Community 16 - "Shared API Helpers"
Cohesion: 0.29
Nodes (3): _load_csv_or_404(), Shared utility functions for route handlers., Load a CSV artifact; raise 404 if not yet created by the pipeline.

### Community 17 - "Pipeline API Routes"
Cohesion: 0.33
Nodes (3): Routes: /health, /pipeline/run, /pipeline/status, Trigger the full pipeline asynchronously., run_pipeline()

### Community 18 - "FastAPI App Entry"
Cohesion: 1.0
Nodes (1): FastAPI application — router wiring only.  All route logic lives in api/routes/.

### Community 19 - "Shared API State"
Cohesion: 1.0
Nodes (1): Shared in-memory state for the single-user local demo.  Imported by pipeline_r

### Community 20 - "CSV Storage Constraint"
Cohesion: 1.0
Nodes (2): CSV-Only Artifact Storage, Rationale No Parquet pyarrow wheels

### Community 21 - "Sparse Series Design"
Cohesion: 1.0
Nodes (2): Sparse Series Fallback, Rationale Sparse Fallback Intentional

### Community 22 - "Single-File Frontend"
Cohesion: 1.0
Nodes (2): Single-File Frontend Architecture, Rationale Single File Frontend Demo

### Community 23 - "Data Quarantine"
Cohesion: 1.0
Nodes (1): Total number of quarantined rows accumulated so far.

### Community 24 - "Backend Documentation"
Cohesion: 1.0
Nodes (1): Backend CLAUDE.md

### Community 25 - "Frontend Documentation"
Cohesion: 1.0
Nodes (1): Frontend CLAUDE.md

### Community 26 - "Python Dependencies"
Cohesion: 1.0
Nodes (1): Backend Requirements

### Community 27 - "Net Amount Sign Rule"
Cohesion: 1.0
Nodes (1): Sign Convention Negative Sale

### Community 28 - "No Future Leakage Rule"
Cohesion: 1.0
Nodes (1): No Future Leakage Rule

### Community 29 - "LLM In-Process Invocation"
Cohesion: 1.0
Nodes (1): LLM In-Process Tool Invocation

## Knowledge Gaps
- **104 isolated node(s):** `Central configuration for the forecasting pipeline. All paths, thresholds, and`, `FastAPI application — router wiring only.  All route logic lives in api/routes/.`, `Shared utility functions for route handlers.`, `Load a CSV artifact; raise 404 if not yet created by the pipeline.`, `LLM Gateway — Multi-provider adapter with tool-calling for Predictify.  Supports` (+99 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `FastAPI App Entry`** (2 nodes): `app.py`, `FastAPI application — router wiring only.  All route logic lives in api/routes/.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Shared API State`** (2 nodes): `state.py`, `Shared in-memory state for the single-user local demo.  Imported by pipeline_r`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `CSV Storage Constraint`** (2 nodes): `CSV-Only Artifact Storage`, `Rationale No Parquet pyarrow wheels`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Sparse Series Design`** (2 nodes): `Sparse Series Fallback`, `Rationale Sparse Fallback Intentional`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Single-File Frontend`** (2 nodes): `Single-File Frontend Architecture`, `Rationale Single File Frontend Demo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Data Quarantine`** (1 nodes): `Total number of quarantined rows accumulated so far.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Backend Documentation`** (1 nodes): `Backend CLAUDE.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Frontend Documentation`** (1 nodes): `Frontend CLAUDE.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Python Dependencies`** (1 nodes): `Backend Requirements`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Net Amount Sign Rule`** (1 nodes): `Sign Convention Negative Sale`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `No Future Leakage Rule`** (1 nodes): `No Future Leakage Rule`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `LLM In-Process Invocation`** (1 nodes): `LLM In-Process Tool Invocation`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BaseForecaster` connect `Model Baselines and Base Classes` to `Ridge Regression Model`, `Config and External Connectors`, `Forecasting Pipeline`, `Lasso and ElasticNet Models`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Why does `Backend services — persistence and other shared infrastructure.` connect `Forecasting Pipeline` to `Model Baselines and Base Classes`, `Data Cleaning Pipeline`, `Config and External Connectors`, `Lasso and ElasticNet Models`, `Ridge Regression Model`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Why does `QuarantineLog` connect `Data Cleaning Pipeline` to `Ridge Regression Model`, `Config and External Connectors`, `Forecasting Pipeline`, `Lasso and ElasticNet Models`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Are the 21 inferred relationships involving `BaseForecaster` (e.g. with `NaiveLastValue` and `NaiveRollingMean`) actually correct?**
  _`BaseForecaster` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `QuarantineLog` (e.g. with `CLI entry point for the Sales Forecasting pipeline.  Usage -----   python ru` and `Print data quality check results.`) actually correct?**
  _`QuarantineLog` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `LassoForecaster` (e.g. with `CLI entry point for the Sales Forecasting pipeline.  Usage -----   python ru` and `Print data quality check results.`) actually correct?**
  _`LassoForecaster` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `RidgeForecaster` (e.g. with `CLI entry point for the Sales Forecasting pipeline.  Usage -----   python ru` and `Print data quality check results.`) actually correct?**
  _`RidgeForecaster` has 6 INFERRED edges - model-reasoned connections that need verification._