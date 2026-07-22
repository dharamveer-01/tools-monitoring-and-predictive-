import os

directories = [
    "alerts",
    "api/routes",
    "api/schemas",
    "api/services",
    "dashboard/frontend/public",
    "dashboard/frontend/src",
    "dashboard/grafana/dashboards",
    "data/processed",
    "data/raw",
    "data/sample_waveforms",
    "data/synthetic_fault_data",
    "database/influxdb",
    "database/timescaledb",
    "deployment/aws",
    "deployment/docker",
    "deployment/kubernetes",
    "deployment/terraform",
    "docs",
    "explainability",
    "feature_engineering",
    "kafka/consumer",
    "kafka/producer",
    "ml_models/anomaly_detection",
    "ml_models/load_balancing",
    "ml_models/prediction",
    "ml_models/saved_models",
    "monitoring/logging",
    "monitoring/prometheus",
    "notebooks",
    "sensors",
    "smart_grid",
    "streaming",
    "testing/integration_tests",
    "testing/load_tests",
    "testing/streaming_tests",
    "testing/unit_tests"
]

files = {
    "alerts/critical_shutdown.py": "# Critical shutdown logic",
    "alerts/email_alert.py": "# Email alert implementation",
    "alerts/slack_alert.py": "# Slack alert implementation",
    "alerts/sms_alert.py": "# SMS alert implementation",
    "api/routes/load_balancing_routes.py": "# Load balancing API endpoints",
    "api/routes/prediction_routes.py": "# Prediction API endpoints",
    "api/routes/telemetry_routes.py": "# Telemetry API endpoints",
    "api/schemas/fault_schema.py": "# Fault schemas",
    "api/schemas/response_schema.py": "# Response schemas",
    "api/schemas/telemetry_schema.py": "# Telemetry schemas",
    "api/services/alert_service.py": "# Alert service logic",
    "api/services/prediction_service.py": "# Prediction service logic",
    "api/services/smart_grid_service.py": "# Smart grid core service",
    "dashboard/frontend/package.json": '{\n  "name": "dashboard",\n  "version": "1.0.0"\n}',
    "database/db_manager.py": "# Database connection manager",
    "database/schema.sql": "-- SQL schema definitions\n",
    "docs/api_documentation.md": "# API Documentation",
    "docs/architecture.md": "# System Architecture",
    "docs/model_documentation.md": "# ML Models Documentation",
    "docs/smart_grid_design.md": "# Smart Grid Design",
    "explainability/feature_importance.py": "# Feature importance using SHAP",
    "explainability/root_cause_engine.py": "# Root cause analysis",
    "explainability/shap_explainer.py": "# SHAP explainer wrapper",
    "feature_engineering/feature_pipeline.py": "# Feature engineering pipeline",
    "feature_engineering/fft_analysis.py": "# Fast Fourier Transform analysis",
    "feature_engineering/harmonic_analysis.py": "# Harmonic analysis logic",
    "kafka/topics.md": "# Kafka Topics configuration",
    "kafka/consumer/alert_consumer.py": "# Kafka alert consumer",
    "kafka/consumer/telemetry_consumer.py": "# Kafka telemetry consumer",
    "kafka/producer/telemetry_producer.py": "# Kafka telemetry producer",
    "ml_models/anomaly_detection/autoencoder.py": "# Autoencoder anomaly model",
    "ml_models/anomaly_detection/isolation_forest.py": "# Isolation forest anomaly model",
    "ml_models/load_balancing/load_optimizer.py": "# Load optimization model",
    "ml_models/prediction/health_score_model.py": "# Health score predictor",
    "monitoring/metrics_collector.py": "# Prometheus metrics collector",
    "sensors/current_sensor.py": "# Current sensor simulation",
    "sensors/thermal_sensor.py": "# Thermal sensor simulation",
    "sensors/voltage_sensor.py": "# Voltage sensor simulation",
    "smart_grid/self_healing_engine.py": "# Self-healing logic",
    "smart_grid/substation_manager.py": "# Substation manager logic",
    "streaming/spark_streaming.py": "# Apache Spark streaming job",
    "requirements.txt": "fastapi\nscikit-learn\npandas\nstreamlit\nuvicorn\nrequests\npydantic\nkafka-python\n"
}

def scaffold():
    base_dir = r"c:\Users\Manish Kudtarkar\industrial-smart-grid-ai"
    
    for d in directories:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
        
    for f, content in files.items():
        path = os.path.join(base_dir, f)
        # only create if doesn't exist to avoid overwriting existing code
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as file:
                file.write(content)
                
    print("Scaffolding complete.")

if __name__ == "__main__":
    scaffold()