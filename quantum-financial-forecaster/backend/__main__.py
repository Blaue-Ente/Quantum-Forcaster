"""Точка на стартиране на FastAPI сървъра."""
import uvicorn

from .api.server import app


def main() -> None:
    uvicorn.run(
        "backend.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
