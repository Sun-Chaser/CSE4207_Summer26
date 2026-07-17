import json
import httpx
from fastapi import HTTPException, APIRouter
from pydantic import BaseModel

router = APIRouter()

class TaskRequest(BaseModel):
    prompt: str

# Dummy function representing your YAML submission logic
def submit_yaml_job(job_type: str, cmd: str):
    # Your Kubernetes / Job orchestration logic goes here
    print(f"Submitting YAML job for {job_type} with command: {cmd}")
    return "job_id_12345"

@router.post("/tasks")
async def submit_task(request: TaskRequest):
    external_service_url = "https://api.your-service.com/analyze" # Replace with actual URL
    
    async with httpx.AsyncClient() as client:
        try:
            # Send the request to the external service
            async with client.stream("POST", external_service_url, json={"prompt": request.prompt}) as response:
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail="External service error")

                # Iterate through the streaming response line by line
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    # Parse the current JSON object in the sequence
                    data = json.loads(line)
                    status = data.get("status")
                    payload = data.get("payload", {})

                    if status == "Valid":
                        job_type = payload.get("job")
                        cmd = payload.get("cmd")
                        
                        # Trigger the YAML job submission
                        job_id = submit_yaml_job(job_type, cmd)
                        
                        return {
                            "status": "success",
                            "message": "Job submitted successfully",
                            "job_id": job_id,
                            "details": payload
                        }
                        
                    elif status == "Invalid":
                        error_msg = payload.get("message", "Unknown validation error.")
                        raise HTTPException(status_code=400, detail=error_msg)
                        
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Service unavailable: {exc}")

    raise HTTPException(status_code=500, detail="Service closed connection without sending a valid response.")