from fastapi import FastAPI

app = FastAPI(title="da5402w")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "hello world"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
