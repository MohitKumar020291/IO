from fastapi import APIRouter, Response


router = APIRouter(prefix="/util")

@router.get("/set_cookie")
async def set_cookie(kv: dict, response: Response):
    key=kv.get("key", None)
    value=kv.get("vaue")
    if key is None:
        ...
    if value is None:
        ...
    response.set_cookie(
        key,
        value,
        httponly=True,
        secure=True,
        samesite="Lax"
    )
    return {"cookie_set_status": True}