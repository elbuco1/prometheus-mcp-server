# Metrics MCP Server

## Features

### Prometheus

This MCP server provides tools to:
- List metrics with filtering and pagination via Grafana’s Prometheus proxy
  - Tool: `list_prometheus_metrics(filter?, limit?, pageToken?)`
  - Deterministic ordering, 200 items/page cap, returns `totalMetricFound`
- Execute queries
  - Tool: `execute_query(query, time?)` (instant PromQL)
  - Tool: `execute_range_query(query, start, end, step)` (range PromQL)
- Inspect metadata and targets
  - Tool: `get_metric_metadata(metric)`
  - Tool: `get_targets()`
- Health check (validates connectivity through Grafana proxy)
  - Tool: `health_check()`

### Graphite

- List metrics with filtering and pagination (cached)
  - Tool: `list_graphite_metrics(filter?, limit?, pageToken?)`
  - Uses Grafana Graphite proxy `metrics/index.json`, deterministic ordering, returns `totalMetricFound`
- Find keys using hierarchical glob patterns (VictoriaMetrics expand)
  - Tool: `find_graphite_keys(pattern, limit?, pageToken?)`
  - Returns paginated `directories` and `metrics` with `totalMetricFound`
- Query time-series data
  - Tool: `query_graphite(targets: List[str], from_?='-5min', until?, maxDataPoints?)`
- Health check (validates Graphite through Grafana proxy)
  - Tool: `health_check()`

### Grafana dashboards

- Create or update a per-user dashboard and append a panel
  - Tool: `create_grafana_dashboard(username, query, datasourceType, legend?, panelTitle?, refresh?='5s', tags?)`
  - Uses/creates folder named after the Linux `username`
  - Targets a dashboard titled `metrics-mcp` in that folder; creates it if missing, otherwise appends a panel
  - `datasourceType`: `prometheus` or `graphite`; sets panel target key (`expr` vs `target`) accordingly
  - Returns `id`, `uid`, relative `url`, absolute `dashboardUrl`, and `folderId`
  - Panel datasource is set by name (e.g., `victoriametrics`, `Graphite-VictoriaMetrics`)
## Local dev

Refer to the ``test_server.py`` script to test your mcp server locally at development time.

## Build the server

```bash
docker build -t metrics-mcp:local .

docker tag metrics-mcp:local 245310300529.dkr.ecr.us-east-1.amazonaws.com/offroad-q3-2025/metrics-mcp:v1

docker push 245310300529.dkr.ecr.us-east-1.amazonaws.com/offroad-q3-2025/metrics-mcp:v1

```

## Use the mcp server with claude code

```bash
claude mcp add metrics-mcp -- docker run -i --rm --network host 245310300529.dkr.ecr.us-east-1.amazonaws.com/offroad-q3-2025/metrics-mcp:v1
```