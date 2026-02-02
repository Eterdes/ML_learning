from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "ml churn service is running"}

#py -m uvicorn api.main:app --reload