"""Global error handlers."""
from fastapi import Request
from fastapi.responses import JSONResponse


async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={"error": str(exc), "code": "VALIDATION_ERROR", "details": {}},
    )


async def not_found_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=404,
        content={"error": str(exc), "code": "NOT_FOUND", "details": {}},
    )
