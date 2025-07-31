#!/usr/bin/env python3
"""
Development runner for Body Tracking AI
Use this for development with auto-reload enabled
"""

import uvicorn
import sys
import os

if __name__ == "__main__":
    # Prevent accidental execution on import (e.g., by Uvicorn reload)
    if sys.argv[0].endswith("run_dev.py"):
        print("🔧 Starting Body Tracking AI in DEVELOPMENT mode")
        print("📝 Auto-reload is ENABLED")
        print("⚠️  Use 'python colab.py' for production mode")
        print("-" * 50)
        try:
            uvicorn.run(
                "fixed_colab:app", 
                host="localhost", 
                port=8000, 
                reload=True,
                reload_dirs=["./"],
                log_level="info"
            )
        except KeyboardInterrupt:
            print("\n👋 Development server stopped")
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            sys.exit(1)
