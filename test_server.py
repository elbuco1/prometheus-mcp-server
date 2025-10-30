import asyncio
from fastmcp import Client

async def main():
    # Connect to your HTTP server (FastMCP uses JSON-RPC over /mcp)
    async with Client("http://localhost:8080/mcp") as client:
        await client.ping()
        result = await client.call_tool("health_check", {})
        print(result)
        print("--------\n")

        # Example: create/update Grafana dashboard panel (Graphite)
        result = await client.call_tool("create_grafana_dashboard", {
            "username": "user-demo",
            "datasourceType": "graphite",
            "query": "alias(summarize(transformNull(qstats.count.model_service.kubernetes.predict.send.by_model_id.3000214.all.count, 0), '1m', 'avg'), \"3000214\")",
            "panelTitle": "Graphite: model 3000214 (1m avg)",
            "legend": "3000214",
            "refresh": "5s",
            "tags": ["graphite", "predictions"]
        })
        print(result)
        print("--------\n")

        # # Example: create/update Grafana dashboard panel (Prometheus)
        # result = await client.call_tool("create_grafana_dashboard", {
        #     "username": "user-demo",
        #     "datasourceType": "prometheus",
        #     "query": "sum by (model_id) (rate(oreo_mlserve_predictions_total[5m]))",
        #     "panelTitle": "Predictions by Model (5m rate)",
        #     "legend": "{{model_id}}",
        #     "refresh": "5s",
        #     "tags": ["api", "predictions"]
        # })
        # print(result)
        # print("--------\n")

        # result = await client.call_tool("list_prometheus_metrics", {"limit": 10})
        # print(result)
        # print("--------\n")

        # result = await client.call_tool("list_prometheus_metrics", {
        #     "limit": 10,
        #     "pageToken": "10",
        # })
        # print(result)
        # print("--------\n")

        # result = await client.call_tool("list_graphite_metrics", {
        #     "limit": 10,
        # })
        # print(result)
        # print("--------\n")

        # result = await client.call_tool("list_graphite_metrics", {
        #     "pageToken": "10",
        #     "limit": 10,
        # })
        # print(result)
        # print("--------\n")

        # # Example Graphite query with aggregation and alias
        # result = await client.call_tool("query_graphite", {
        #     "targets": [
        #         "alias(summarize(transformNull(qstats.count.model_service.kubernetes.predict.send.by_model_id.3000214.all.count, 0), '1m', 'avg'), \"3000214\")"
        #     ],
        #     "from_": "-15min",
        #     "maxDataPoints": 600,
        # })
        # print(result)
        # print("--------\n")

        # Example Graphite key search (glob pattern) with pagination
        # result = await client.call_tool("find_graphite_keys", {
        #     "pattern": "qstats.count.model_service.kubernetes.predict.send.by_model_id.*",
        #     "limit": 5
        # })
        # print(result)
        # print("--------\n")

        # # Next page using nextPageToken
        # if result and hasattr(result, "data") and isinstance(result.data, dict) and result.data.get("nextPageToken"):
        #     next_token = result.data["nextPageToken"]
        #     result2 = await client.call_tool("find_graphite_keys", {
        #         "pattern": "qstats.count.model_service.kubernetes.predict.send.by_model_id.*",
        #         "limit": 5,
        #         "pageToken": next_token
        #     })
        #     print(result2)
        #     print("--------\n")


asyncio.run(main())