#!/usr/bin/env python
import sys
import dotenv
from prometheus_mcp_server.server import mcp, config, TransportType
from prometheus_mcp_server.logging_config import setup_logging

# Initialize structured logging
logger = setup_logging()

def setup_environment():
    # MCP Server configuration validation
    mcp_config = config.mcp_server_config
    if mcp_config:
        if str(mcp_config.mcp_server_transport).lower() not in TransportType.values():
            logger.error(
                "Invalid mcp transport",
                error="PROMETHEUS_MCP_SERVER_TRANSPORT environment variable is invalid",
                suggestion="Please define one of these acceptable transports (http/sse/stdio)",
                example="http"
            )
            return False

        try:
            if mcp_config.mcp_bind_port:
                int(mcp_config.mcp_bind_port)
        except (TypeError, ValueError):
            logger.error(
                "Invalid mcp port",
                error="PROMETHEUS_MCP_BIND_PORT environment variable is invalid",
                suggestion="Please define an integer",
                example="8080"
            )
            return False
    
    logger.info(
        "Server configuration validated",
        prometheus_url=config.prometheus_url,
        graphite_url=config.graphite_url,
        grafana_dashboard_url=config.grafana_dashboard_url,
        mcp_server_transport=config.mcp_server_config.mcp_server_transport,
        mcp_bind_host=config.mcp_server_config.mcp_bind_host,
        mcp_bind_port=config.mcp_server_config.mcp_bind_port,
    )
    
    return True

def run_server():
    """Main entry point for the Prometheus MCP Server"""
    # Setup environment
    if not setup_environment():
        logger.error("Environment setup failed, exiting")
        sys.exit(1)
    
    mcp_config = config.mcp_server_config
    transport = mcp_config.mcp_server_transport

    http_transports = [TransportType.HTTP.value, TransportType.SSE.value]
    if transport in http_transports:
        mcp.run(transport=transport, host=mcp_config.mcp_bind_host, port=mcp_config.mcp_bind_port)
        logger.info("Starting Prometheus MCP Server", 
                transport=transport, 
                host=mcp_config.mcp_bind_host,
                port=mcp_config.mcp_bind_port)
    else:
        mcp.run(transport=transport)
        logger.info("Starting Prometheus MCP Server", transport=transport)

if __name__ == "__main__":
    run_server()
