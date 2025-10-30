#!/usr/bin/env python

import os
import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import time
from datetime import datetime, timedelta
from enum import Enum

import dotenv
import requests
from fastmcp import FastMCP
from prometheus_mcp_server.logging_config import get_logger

GRAFANA_BASE_URL = "https://grafana.quora.net/api"
GRAFANA_DATA_SOURCE_URL = GRAFANA_BASE_URL + "/datasources/proxy/{data_source_id}"
GRAFANA_DASHBOARD_URL = f"{GRAFANA_BASE_URL}/dashboards/db"

class DataSources(Enum):
    PROMETHEUS = 14
    GRAPHITE = 17

class TransportType(str, Enum):
    """Supported MCP server transport types."""

    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"

    @classmethod
    def values(cls) -> list[str]:
        """Get all valid transport values."""
        return [transport.value for transport in cls]

@dataclass
class MCPServerConfig:
    """Global Configuration for MCP."""
    mcp_server_transport: TransportType = None
    mcp_bind_host: str = None
    mcp_bind_port: int = None

    def __post_init__(self):
        """Validate mcp configuration."""
        if not self.mcp_server_transport:
            raise ValueError("MCP SERVER TRANSPORT is required")
        if not self.mcp_bind_host:
            raise ValueError(f"MCP BIND HOST is required")
        if not self.mcp_bind_port:
            raise ValueError(f"MCP BIND PORT is required")

@dataclass
class ServerConfig:
    url_ssl_verify: bool
    prometheus_url: str
    graphite_url: str
    grafana_dashboard_url: str
    mcp_server_config: Optional[MCPServerConfig] = None

config = ServerConfig(
    url_ssl_verify=os.environ.get("PROMETHEUS_URL_SSL_VERIFY", True),
    prometheus_url=GRAFANA_DATA_SOURCE_URL.format(data_source_id=DataSources.PROMETHEUS.value),
    graphite_url=GRAFANA_DATA_SOURCE_URL.format(data_source_id=DataSources.GRAPHITE.value),
    grafana_dashboard_url=GRAFANA_DASHBOARD_URL,
    mcp_server_config=MCPServerConfig(
        mcp_server_transport=os.environ.get("METRICS_MCP_SERVER_TRANSPORT", "stdio").lower(),
        mcp_bind_host=os.environ.get("METRICS_MCP_BIND_HOST", "127.0.0.1"),
        mcp_bind_port=int(os.environ.get("METRICS_MCP_BIND_PORT", "8080"))
    )
)

mcp = FastMCP("Metrics MCP")

# Cache for metrics list to improve completion performance
_metrics_cache = {"data": None, "timestamp": 0}
_CACHE_TTL = 300  # 5 minutes

# Cache for Graphite metrics index to avoid repeated large downloads
_graphite_metrics_cache = {"data": None, "timestamp": 0}

# Shared pagination cap for metrics listings (Prometheus/Graphite)
METRICS_PAGE_MAX = 200

def _get_cached_list(cache_store: Dict[str, Any], fetch_fn, cache_name: str) -> List[str]:
    """Generic cache wrapper for list endpoints with TTL and deterministic sorting.

    Args:
        cache_store: Dict with keys "data" and "timestamp".
        fetch_fn: Function that returns a List[str] when called.
        cache_name: Name used for logs.

    Returns:
        List[str]: Cached or freshly fetched list; sorted alphabetically when refreshed.
    """
    current_time = time.time()

    if cache_store["data"] is not None and (current_time - cache_store["timestamp"]) < _CACHE_TTL:
        logger.debug("Using cached list", cache=cache_name, cache_age=current_time - cache_store["timestamp"])
        return cache_store["data"]

    try:
        data = fetch_fn()
        # Ensure deterministic ordering for stable pagination
        data = sorted(data)
        cache_store["data"] = data
        cache_store["timestamp"] = current_time
        logger.debug("Refreshed cached list", cache=cache_name, item_count=len(data))
        return data
    except Exception as e:
        logger.error("Failed to refresh cached list", cache=cache_name, error=str(e))
        return cache_store["data"] if cache_store["data"] is not None else []

# Get logger instance
logger = get_logger()

# Health check tool for Docker containers and monitoring
@mcp.tool(
    description="Health check endpoint for container monitoring and status verification of underlying Grafana, Prometheus, and Graphite services",
    annotations={
        "title": "Health Check",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def health_check() -> Dict[str, Any]:
    """Return health status of the MCP server and Prometheus and Graphite connections through Grafana.
    
    Returns:
        Health status including service information, configuration, and connectivity
    """
    try:
        health_status = {
            "status": "healthy",
            "service": "metrics-mcp-server",
            "timestamp": datetime.utcnow().isoformat(),
            "transport": config.mcp_server_config.mcp_server_transport if config.mcp_server_config else "stdio",
            "prometheus": {
                "status": "unknown",
            },
            "graphite": {
                "status": "unknown",
            },
            "grafana": {
                "status": "unknown",
            },
        }
        
        # Test Grafana API health
        try:
            resp = requests.get(f"{GRAFANA_BASE_URL}/health", timeout=10)
            resp.raise_for_status()
            health_status["grafana"]["status"] = "healthy"
        except Exception as e:
            health_status["grafana"]["status"] = "unhealthy"
            health_status["grafana"]["error"] = str(e)
            health_status["status"] = "degraded"

        # Test Prometheus connectivity (through Grafana datasource proxy)
        try:
            # Quick connectivity test
            make_prometheus_request("query", params={"query": "up", "time": str(int(time.time()))})
            health_status["prometheus"]["status"] = "healthy"
        except Exception as e:
            health_status["prometheus"]["status"] = "unhealthy"
            health_status["prometheus"]["error"] = str(e)
            health_status["status"] = "degraded"

        # Test Graphite connectivity (through Grafana datasource proxy)
        try:
            graphite_base = config.graphite_url.rstrip("/")
            resp = requests.get(
                f"{graphite_base}/metrics/find",
                params={"query": "*"},
                timeout=10,
            )
            resp.raise_for_status()
            health_status["graphite"]["status"] = "healthy"
        except Exception as e:
            health_status["graphite"]["status"] = "unhealthy"
            health_status["graphite"]["error"] = str(e)
            health_status["status"] = "degraded"
            
        logger.info("Health check completed", status=health_status["status"])
        return health_status
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "service": "prometheus-mcp-server",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


def make_prometheus_request(endpoint, params=None):
    """Make a request to the Prometheus API with proper authentication and headers."""

    if not config.url_ssl_verify:
        logger.warning("SSL certificate verification is disabled. This is insecure and should not be used in production environments.", endpoint=endpoint)

    url = f"{config.prometheus_url.rstrip('/')}/api/v1/{endpoint}"
    headers = {}

    try:
        logger.debug("Making Prometheus API request", endpoint=endpoint, url=url, params=params)
        
        # Make the request with appropriate headers
        response = requests.get(url, params=params, headers=headers, verify=config.url_ssl_verify)
        
        response.raise_for_status()
        result = response.json()
        
        if result["status"] != "success":
            error_msg = result.get('error', 'Unknown error')
            logger.error("Prometheus API returned error", endpoint=endpoint, error=error_msg, status=result["status"])
            raise ValueError(f"Prometheus API error: {error_msg}")
        
        logger.debug("Prometheus API request successful", endpoint=endpoint)
        return result["data"]
    
    except requests.exceptions.RequestException as e:
        logger.error("HTTP request to Prometheus failed", endpoint=endpoint, url=url, error=str(e), error_type=type(e).__name__)
        raise
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Prometheus response as JSON", endpoint=endpoint, url=url, error=str(e))
        raise ValueError(f"Invalid JSON response from Prometheus: {str(e)}")
    except Exception as e:
        logger.error("Unexpected error during Prometheus request", endpoint=endpoint, url=url, error=str(e), error_type=type(e).__name__)
        raise

def get_cached_prometheus_metrics() -> List[str]:
    """Get metrics list with caching to improve completion performance.

    This helper function is available for future completion support when
    FastMCP implements the completion capability. For now, it can be used
    internally to optimize repeated metric list requests.
    """
    current_time = time.time()

    # Check if cache is valid
    if _metrics_cache["data"] is not None and (current_time - _metrics_cache["timestamp"]) < _CACHE_TTL:
        logger.debug("Using cached metrics list", cache_age=current_time - _metrics_cache["timestamp"])
        return _metrics_cache["data"]

    # Fetch fresh metrics
    return _get_cached_list(
        _metrics_cache,
        lambda: make_prometheus_request("label/__name__/values"),
        cache_name="prometheus_metrics",
    )

def get_cached_graphite_metrics() -> List[str]:
    """Fetch Graphite metrics index.json with caching.

    Returns the full list of metric paths (strings).
    """
    current_time = time.time()

    def _fetch_graphite_index() -> List[str]:
        url = f"{config.graphite_url.rstrip('/')}/metrics/index.json"
        logger.debug("Fetching Graphite metrics index", url=url)
        resp = requests.get(url, timeout=30, verify=config.url_ssl_verify)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise ValueError(f"Unexpected Graphite index format: {type(data).__name__}")
        return data

    return _get_cached_list(
        _graphite_metrics_cache,
        _fetch_graphite_index,
        cache_name="graphite_metrics_index",
    )

@mcp.tool(
    description="Execute a PromQL instant query against Prometheus",
    annotations={
        "title": "Execute PromQL Query",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def execute_query(query: str, time: Optional[str] = None) -> Dict[str, Any]:
    """Execute an instant query against Prometheus.
    
    Args:
        query: PromQL query string
        time: Optional RFC3339 or Unix timestamp (default: current time)
        
    Returns:
        Query result with type (vector, matrix, scalar, string) and values
    """
    params = {"query": query}
    if time:
        params["time"] = time
    
    logger.info("Executing instant query", query=query, time=time)
    data = make_prometheus_request("query", params=params)

    result = {
        "resultType": data["resultType"],
        "result": data["result"],
    }

    logger.info("Instant query completed",
                query=query,
                result_type=data["resultType"],
                result_count=len(data["result"]) if isinstance(data["result"], list) else 1)

    return result

@mcp.tool(
    description="List all available metrics in Graphite (via Grafana proxy) with optional filtering and pagination",
    annotations={
        "title": "List Graphite Metrics",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def list_graphite_metrics(filter: Optional[str] = None, limit: int = 200, pageToken: Optional[str] = None, ctx=None) -> Dict[str, Any]:
    """List Graphite metrics via Grafana's datasource proxy with client-side filtering and stable pagination.

    Behavior:
        - Downloads Graphite's full metric index from `{graphite_url}/metrics/index.json` (through Grafana proxy) and caches it for 5 minutes.
        - Metrics are sorted alphabetically before caching to guarantee stable, deterministic pagination across calls.
        - Optional substring `filter` is applied client-side to the cached list (case-sensitive containment).
        - Pagination uses a zero-based offset encoded as a string `pageToken` and a `limit` cap.

    Args:
        filter: Optional substring to match metric paths (e.g., "cpu."), case-sensitive.
        limit: Max metrics to return in this page (default 200; clamped to 1..METRICS_PAGE_MAX).
        pageToken: Opaque cursor representing the current offset (stringified int). Omit or "0" for the first page.

    Returns:
        Dict with shape:
            {
                "source": "graphite",
                "metrics": [str, ...],           # alphabetically sorted page of metric paths
                "nextPageToken": str | None      # pass to next call; None when no more results
            }

    Notes for LLMs:
        - For large listings, iterate using `nextPageToken` until it returns null.
        - Use `filter` to narrow results and reduce token/latency costs.
        - Page sizes above METRICS_PAGE_MAX are automatically reduced to METRICS_PAGE_MAX.
    """
    # Normalize arguments
    limit = max(1, min(limit, METRICS_PAGE_MAX))

    try:
        if ctx:
            await ctx.report_progress(progress=0, total=100, message="Loading Graphite metrics index...")

        all_metrics = get_cached_graphite_metrics()

        if filter:
            metrics = [m for m in all_metrics if filter in m]
        else:
            metrics = all_metrics

        # Pagination by offset encoded as pageToken
        try:
            start = int(pageToken) if pageToken is not None else 0
        except ValueError:
            start = 0
        end = start + limit

        page = metrics[start:end]
        next_token = str(end) if end < len(metrics) else None

        if ctx:
            await ctx.report_progress(progress=100, total=100, message=f"Returned {len(page)} metrics")

        logger.info("Graphite metrics listed", filter=filter, page_start=start, page_size=len(page), next_token_present=bool(next_token))
        return {"source": "graphite", "metrics": page, "nextPageToken": next_token}

    except Exception as e:
        logger.error("Failed to list Graphite metrics", error=str(e))
        return {"source": "graphite", "metrics": [], "nextPageToken": None, "error": str(e)}

@mcp.tool(
    description="Execute a PromQL range query with start time, end time, and step interval",
    annotations={
        "title": "Execute PromQL Range Query",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def execute_range_query(query: str, start: str, end: str, step: str, ctx=None) -> Dict[str, Any]:
    """Execute a range query against Prometheus.
    
    Args:
        query: PromQL query string
        start: Start time as RFC3339 or Unix timestamp
        end: End time as RFC3339 or Unix timestamp
        step: Query resolution step width (e.g., '15s', '1m', '1h')
        
    Returns:
        Range query result with type (usually matrix) and values over time
    """
    params = {
        "query": query,
        "start": start,
        "end": end,
        "step": step
    }

    logger.info("Executing range query", query=query, start=start, end=end, step=step)

    # Report progress if context available
    if ctx:
        await ctx.report_progress(progress=0, total=100, message="Initiating range query...")

    data = make_prometheus_request("query_range", params=params)

    # Report progress
    if ctx:
        await ctx.report_progress(progress=50, total=100, message="Processing query results...")

    result = {
        "resultType": data["resultType"],
        "result": data["result"],
    }

    # Report completion
    if ctx:
        await ctx.report_progress(progress=100, total=100, message="Range query completed")

    logger.info("Range query completed",
                query=query,
                result_type=data["resultType"],
                result_count=len(data["result"]) if isinstance(data["result"], list) else 1)

    return result

@mcp.tool(
    description="List all available metrics in Prometheus (via Grafana proxy) with optional filtering and pagination",
    annotations={
        "title": "List Prometheus Metrics",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def list_prometheus_metrics(filter: Optional[str] = None, limit: int = 200, pageToken: Optional[str] = None, ctx=None) -> Dict[str, Any]:
    """List Prometheus metric names via Grafana's datasource proxy with client-side filtering and stable pagination.

    Behavior:
        - Loads the full metric name list from `/api/v1/label/__name__/values` (via Grafana proxy) and caches it for 5 minutes.
        - The cached list is sorted alphabetically for deterministic ordering and stable pagination.
        - Optional substring `filter` is applied client-side (case-sensitive) to narrow results.
        - Pagination uses a zero-based offset encoded as a string `pageToken` and a `limit` cap.

    Args:
        filter: Optional substring to match metric names (e.g., "http_"), case-sensitive.
        limit: Max metrics to return in this page (default 200; clamped to 1..METRICS_PAGE_MAX).
        pageToken: Opaque cursor representing the current offset (stringified int). Omit or "0" for the first page.

    Returns:
        Dict with shape:
            {
                "source": "prometheus",
                "metrics": [str, ...],          # alphabetically sorted page of metric names
                "nextPageToken": str | None     # pass to next call; None when no more results
            }
    """
    logger.info("Listing available Prometheus metrics", filter=filter, limit=limit, page_token=pageToken)

    # Report progress if context available
    if ctx:
        await ctx.report_progress(progress=0, total=100, message="Fetching Prometheus metrics list...")

    # Clamp limit
    limit = max(1, min(limit, METRICS_PAGE_MAX))

    all_metrics = get_cached_prometheus_metrics()

    if filter:
        metrics = [m for m in all_metrics if filter in m]
    else:
        metrics = all_metrics

    # Pagination by offset encoded as pageToken
    try:
        start = int(pageToken) if pageToken is not None else 0
    except ValueError:
        start = 0
    end = start + limit

    page = metrics[start:end]
    next_token = str(end) if end < len(metrics) else None

    if ctx:
        await ctx.report_progress(progress=100, total=100, message=f"Returned {len(page)} metrics")

    logger.info("Prometheus metrics listed", page_start=start, page_size=len(page), next_token_present=bool(next_token))
    return {"source": "prometheus", "metrics": page, "nextPageToken": next_token}

@mcp.tool(
    description="Get metadata for a specific metric",
    annotations={
        "title": "Get Metric Metadata",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def get_metric_metadata(metric: str) -> List[Dict[str, Any]]:
    """Get metadata about a specific metric.
    
    Args:
        metric: The name of the metric to retrieve metadata for
        
    Returns:
        List of metadata entries for the metric
    """
    logger.info("Retrieving metric metadata", metric=metric)
    params = {"metric": metric}
    data = make_prometheus_request("metadata", params=params)
    if "metadata" in data:
        metadata = data["metadata"]
    else:
        metadata = data["data"]
    logger.info("Metric metadata retrieved", metric=metric, metadata_count=len(metadata))
    return metadata

@mcp.tool(
    description="Get information about all scrape targets",
    annotations={
        "title": "Get Scrape Targets",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def get_targets() -> Dict[str, List[Dict[str, Any]]]:
    """Get information about all Prometheus scrape targets.
    
    Returns:
        Dictionary with active and dropped targets information
    """
    logger.info("Retrieving scrape targets information")
    data = make_prometheus_request("targets")
    
    result = {
        "activeTargets": data["activeTargets"],
        "droppedTargets": data["droppedTargets"]
    }
    
    logger.info("Scrape targets retrieved", 
                active_targets=len(data["activeTargets"]), 
                dropped_targets=len(data["droppedTargets"]))
    
    return result

if __name__ == "__main__":
    logger.info("Starting Prometheus MCP Server", mode="direct")
    mcp.run()
