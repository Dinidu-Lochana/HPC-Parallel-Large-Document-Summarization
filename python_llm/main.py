from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import os
import shutil
import uuid
import tempfile
import subprocess
from parallel_summarizer import parallel_summarize_document
from summarizer import summarize_document

app = FastAPI(
    title="HPC Parallel Document Summarization API",
    description="API for parallel document summarization using Gemini",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the HPC Parallel Document Summarization API"}

@app.post("/summarize")
async def summarize(
    file: UploadFile = File(...),
    topic: str = Form(""),
    max_workers: int = Form(5)
):
    """
    Summarize a document using parallel processing.
    """
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    
    # Create a temporary directory
    temp_dir = tempfile.gettempdir()
    unique_id = str(uuid.uuid4())
    temp_file_path = os.path.join(temp_dir, f"{unique_id}_{file.filename}")
    
    try:
        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run parallel summarization
        summary = parallel_summarize_document(temp_file_path, topic=topic, max_workers=max_workers)
        
        return JSONResponse(content={
            "filename": file.filename,
            "topic": topic,
            "summary": summary
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/summarize-stream")
async def summarize_stream(
    file: UploadFile = File(...),
    topic: str = Form("")
):
    """
    Summarize a document using streaming (chunk by chunk).
    Uses the non-parallel streaming generator.
    """
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    
    temp_dir = tempfile.gettempdir()
    unique_id = str(uuid.uuid4())
    temp_file_path = os.path.join(temp_dir, f"{unique_id}_{file.filename}")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
            
    def event_generator():
        try:
            dummy_file = open(temp_file_path, 'rb')
            
            for token in summarize_document(dummy_file, topic=topic, file_name=file.filename):
                yield token
                
            dummy_file.close()
        except Exception as e:
            yield f"\n[Error: {str(e)}]"
        finally:
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass
                
    return StreamingResponse(event_generator(), media_type="text/plain")

@app.post("/summarize-cuda")
async def summarize_cuda(
    file: UploadFile = File(...),
    topic: str = Form("")
):
    """
    Summarize a document using the CUDA C++ binary.
    """
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    
    temp_dir = tempfile.gettempdir()
    unique_id = str(uuid.uuid4())
    temp_file_path = os.path.join(temp_dir, f"{unique_id}_{file.filename}")
    txt_file_path = os.path.join(temp_dir, f"{unique_id}.txt")
    
    try:
        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # PDF, extract text first for the CUDA binary
        if file.filename.lower().endswith('.pdf'):
            from summarizer import read_pdf
            with open(temp_file_path, "rb") as pdf_file:
                extracted_text = read_pdf(pdf_file)
            with open(txt_file_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(extracted_text)
            input_for_cuda = txt_file_path
        else:
            input_for_cuda = temp_file_path

        # Path to the CUDA executable
        base_dir = os.path.dirname(os.path.dirname(__file__))
        cuda_exe = os.path.join(base_dir, "cuda", "cuda_summarizer")
        
        # Add .exe for Windows
        is_wsl = False
        if not os.path.exists(cuda_exe):
            if os.path.exists(cuda_exe + ".exe"):
                cuda_exe += ".exe"
        elif os.name == 'nt' and not cuda_exe.endswith('.exe'):
            is_wsl = True
            
        if not os.path.exists(cuda_exe):
            raise HTTPException(status_code=500, detail=f"CUDA executable not found at {cuda_exe}. Please compile it first.")
            
        # Run CUDA subprocess
        if is_wsl:
            # Convert Windows paths to WSL paths (e.g. C:\Temp\a.txt -> /mnt/c/Temp/a.txt)
            wsl_input = input_for_cuda.replace('\\', '/')
            if len(wsl_input) > 2 and wsl_input[1:3] == ':/':
                wsl_input = f"/mnt/{wsl_input[0].lower()}/{wsl_input[3:]}"
                
            wsl_exe = cuda_exe.replace('\\', '/')
            if len(wsl_exe) > 2 and wsl_exe[1:3] == ':/':
                wsl_exe = f"/mnt/{wsl_exe[0].lower()}/{wsl_exe[3:]}"
                
            cmd = ["wsl", wsl_exe, wsl_input]
        else:
            cmd = [cuda_exe, input_for_cuda]
            
        if topic:
            cmd.append(topic)
            
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"CUDA execution failed: {process.stderr}\n\nSTDOUT: {process.stdout}")
            
        output_text = process.stdout
        
        print("\n--- CUDA EXECUTION LOGS ---")
        print(output_text)
        print("---------------------------\n")
        
        marker = "FINAL SUMMARY\n=================================================\n"
        summary_start_idx = output_text.find(marker)
        
        if summary_start_idx != -1:
            summary_str = output_text[summary_start_idx + len(marker):].strip()
            end_idx = summary_str.rfind("=================================================")
            if end_idx != -1:
                summary_str = summary_str[:end_idx].strip()
        else:
            summary_str = output_text
            
        return JSONResponse(content={
            "filename": file.filename,
            "topic": topic,
            "summary": summary_str,
            "logs": output_text  
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(txt_file_path):
            os.remove(txt_file_path)
