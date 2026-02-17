#!/usr/bin/env python
"""
Launch the ProToPhen REST API server.

Usage::

    # Minimal (requires checkpoint):
    python scripts/serve.py --checkpoint checkpoints/best.pt

    # Full options:
    python scripts/serve.py \\
        --checkpoint checkpoints/best.pt \\
        --config configs/deployment.yaml \\
        --host 0.0.0.0 \\
        --port 8000 \\
        --device cuda \\
        --workers 1 \\
        --reload

    # From model registry (serves production model):
    python scripts/serve.py --registry ./model_registry
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the ProToPhen prediction API server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model source (one of these is required)
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pt model checkpoint.",
    )
    src.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Path to model registry directory (serves the production model).",
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to deployment.yaml configuration file.",
    )

    # Server
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn workers.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload.")

    # Pipeline overrides
    parser.add_argument("--device", type=str, default=None, help="Device override.")
    parser.add_argument("--esm-model", type=str, default=None, help="ESM-2 model override.")

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level override.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure FastAPI + Uvicorn are installed
    try:
        import uvicorn
    except ImportError:
        print(
            "ERROR: uvicorn is required.  Install with:\n"
            "  pip install 'protophen[serving]'\n"
            "  # or: pip install uvicorn fastapi",
            file=sys.stderr,
        )
        sys.exit(1)

    from protophen.serving.api import create_app
    from protophen.serving.pipeline import PipelineConfig
    from protophen.serving.monitoring import MonitoringConfig
    from protophen.serving.registry import ModelRegistry, RegistryConfig
    from protophen.utils.logging import logger, setup_logging

    # ----- Load YAML config if provided -----
    pipeline_kwargs: dict = {}
    monitoring_config = MonitoringConfig()
    registry_dir: str | None = None
    feedback_dir: str | None = None
    api_host = "0.0.0.0"
    api_port = 8000
    api_workers = 1
    log_level = "INFO"
    log_file = str | None = None
    uvicorn_log_level = "info"

    if args.config is not None:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        if "pipeline" in cfg:
            pipeline_kwargs.update(cfg["pipeline"])
        if "monitoring" in cfg:
            monitoring_config = MonitoringConfig(**cfg["monitoring"])
        if "registry" in cfg:
            registry_dir = cfg["registry"].get("registry_dir", registry_dir)
        if "feedback" in cfg:
            feedback_dir = cfg["feedback"].get("persist_dir", feedback_dir)
        if "logging" in cfg:
            log_level = cfg["logging"].get("level", log_level)
            log_file = cfg["logging"].get("log_file", log_file)
        if "api" in cfg:
            api_host = cfg["api"].get("host", api_host)
            api_port = cfg["api"].get("port", api_port)
            api_workers = cfg["api"].get("workers", api_workers)
            uvicorn_log_level = cfg["api"].get("log_level", uvicorn_log_level)
            
    # ----- CLI overrides -----
    if args.host is not None:
        api_host = args.host
    if args.port is not None:
        api_port = args.port
    if args.workers is not None:
        api_workers = args.workers
    if args.log_level is not None:
        log_level = args.log_level
        uvicorn_log_level = log_level.lower()
        
    # ----- Configure logging -----
    setup_logging(level=log_level, log_file=log_file)

    # ----- Resolve checkpoint -----
    checkpoint_path = args.checkpoint

    if args.registry is not None:
        registry_dir = args.registry
        registry = ModelRegistry(config=RegistryConfig(registry_dir=args.registry))
        checkpoint_path = registry.get_production_checkpoint()
        if checkpoint_path is None:
            logger.error(
                f"No production model found in registry '{args.registry}'.\n"
                f"Register and promote a model first:\n"
                f"  from protophen.serving.registry import ModelRegistry\n"
                f"  reg = ModelRegistry('./model_registry')\n"
                f"  reg.register('checkpoints/best.pt', version='v1')\n"
                f"  reg.set_stage('v1', 'production')",
            )
            sys.exit(1)
        logger.info(f"Resolved checkpoint from registry: {checkpoint_path}") 

    if checkpoint_path is None and pipeline_kwargs.get("checkpoint_path") is None:
        logger.warning(
            "No checkpoint provided.  The server will start but "
            "/predict will return 503 until a model is loaded.",
        )

    # ----- CLI overrides -----
    if args.device is not None:
        pipeline_kwargs["device"] = args.device
    if args.esm_model is not None:
        pipeline_kwargs["esm_model_name"] = args.esm_model

    # ----- Build config objects -----
    pipeline_config = PipelineConfig(**pipeline_kwargs)

    # ----- Create app -----
    app = create_app(
        checkpoint_path=checkpoint_path,
        pipeline_config=pipeline_config,
        monitoring_config=monitoring_config,
        registry_dir=registry_dir,
        feedback_dir=feedback_dir,
    )

    # ----- Launch -----
    logger.info(
        f"Starting ProToPhen API on {api_host}:{api_port}"
        f"(workers={api_workers}, reload={args.reload}, log_level={log_level})"
    )
    
    uvicorn.run(
        app,
        host=api_host,
        port=api_port,
        workers=api_workers if not args.reload else 1,
        reload=args.reload,
        log_level=uvicorn_log_level,
    )


if __name__ == "__main__":
    main()