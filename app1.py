from fastapi import FastAPI, Request, Form

app = FastAPI()

@app.get("/")
async def read_root():
    sd = "Welcome to the FastAPI application!"
    return {"message": sd}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)