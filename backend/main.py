from fastapi import FastAPI
from pydantic import BaseModel
from multi_tool_agent.agent import generate_report

app = FastAPI()

class UserInput(BaseModel):
    data: str

@app.post("/generate_report")
async def generate_report_api(user_input: UserInput):
    # Call your agent's generate_report function with the user input data
    result = generate_report(user_input.data)
    return result

# Optional: a simple root endpoint for quick server health check
@app.get("/")
def root():
    return {"message": "Ayurveda report agent is running"}
