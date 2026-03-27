import os
import shutil
import subprocess

base_dir = r"c:\Users\Islab\Desktop\islab_code\ELXGB"
docs_dir = os.path.join(base_dir, "docs")
agents_dir = os.path.join(base_dir, "agents")

os.makedirs(docs_dir, exist_ok=True)
os.makedirs(agents_dir, exist_ok=True)

# Copy artifacts
brain_dir = r"C:\Users\Islab\.gemini\antigravity\brain\a7950c65-d49d-46c4-acaf-641207632811"
try:
    shutil.copy(os.path.join(brain_dir, "implementation_plan.md"), os.path.join(docs_dir, "implementation_plan.md"))
    shutil.copy(os.path.join(brain_dir, "task.md"), os.path.join(docs_dir, "task.md"))
    print("Copied plan and task files to docs/")
except Exception as e:
    print(f"Error copying files: {e}")

# Create Agents
agent_files = {
    "ArchitectureAgent.md": "# Architecture Agent\n자역할: 논문의 훈련/추론 아키텍처를 철저히 분석하고 시각화(Mermaid)하여 사용자 검토를 진행하는 에이전트.",
    "VerificationAgent.md": "# Verification Agent\n역할: 잔잔바리 스텝 구현 시, 논문의 수식(암호화, DP 등)과 일치하는지 코드 무결성을 검증하고 사용자 승인을 구하는 에이전트."
}

for name, content in agent_files.items():
    with open(os.path.join(agents_dir, name), "w", encoding="utf-8") as f:
        f.write(content)

# Attempt to extract PDF text
pdf_path = os.path.join(base_dir, "ELXGB_An_Efficient_and_Privacy-Preserving_XGBoost_for_Vertical_Federated_Learning (1).pdf")
try:
    import fitz # PyMuPDF
    doc = fitz.open(pdf_path)
    text = ""
    for i in range(min(15, len(doc))):
        text += doc[i].get_text()
    
    with open(os.path.join(base_dir, "paper_text.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print("Extracted PDF text using PyMuPDF")
except ImportError:
    print("PyMuPDF not installed, trying PyPDF2...")
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(pdf_path)
        text = ""
        for i in range(min(15, len(reader.pages))):
            text += reader.pages[i].extract_text()
        with open(os.path.join(base_dir, "paper_text.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        print("Extracted PDF text using PyPDF2")
    except Exception as e:
        print(f"Extraction failed: {e}")
