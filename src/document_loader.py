import csv
import json
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, BSHTMLLoader
from langchain_core.documents import Document


def load_documents(data_dir: str = "data/raw") -> List[Document]:
    data_path = Path(data_dir)
    documents = []
    for file_path in data_path.rglob("*"):
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".pdf":
                docs = PyPDFLoader(str(file_path)).load()
            elif suffix in (".docx", ".doc"):
                docs = UnstructuredWordDocumentLoader(str(file_path)).load()
            elif suffix == ".txt":
                text = file_path.read_text(encoding="utf-8")
                docs = [Document(page_content=text, metadata={"source": file_path.name})]
            elif suffix == ".html":
                docs = BSHTMLLoader(str(file_path)).load()
            elif suffix == ".csv":
                docs = _load_csv(file_path)
            elif suffix == ".json":
                docs = _load_json(file_path)
            else:
                continue
            documents.extend(docs)
        except Exception as e:
            print(f"  Warning: could not load {file_path.name}: {e}")
    return documents


def _load_csv(file_path: Path) -> List[Document]:
    docs = []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            content = "\n".join(f"{k}: {v}" for k, v in row.items())
            docs.append(Document(page_content=content, metadata={"source": file_path.name, "row": i}))
    return docs


def _load_json(file_path: Path) -> List[Document]:
    data = json.loads(file_path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else [data]
    return [
        Document(page_content=json.dumps(item, indent=2), metadata={"source": file_path.name, "index": i})
        for i, item in enumerate(items)
    ]

SAMPLE_DOCUMENTS = [
    {
        "content": """
        Healthcare Benefits Summary 2026
        
        Preventive Care: Annual checkups, vaccinations, and screenings are covered 100%
        without copay.
        
        Prescription Drugs: Generic drugs: $10 copay. Brand-name drugs: $35 copay.
        Specialty drugs: $75 copay.
        
        Mental Health: 20 therapy sessions per year covered at $20 copay. After that,
        subject to deductible.
        
        Emergency Room: $150 copay, waived if admitted to hospital.
        
        Out-of-Pocket Maximum: $3,000 individual, $6,000 family.
        """,
        "source": "benefits_summary.txt"
    },
    {
        "content": """
        PTO Policy for Healthcare Employees
        
        Accrual Rates:
        - 0-2 years: 15 days per year
        - 2-5 years: 20 days per year
        - 5+ years: 25 days per year
        
        Sick Leave: 10 days per year, non-accruing.
        
        Carryover: Up to 5 days can be carried over to next year.
        Unused days beyond 5 are forfeited on Jan 1.
        """,
        "source": "pto_policy.txt"
    },
    {
        "content": """
        Diabetes Management Program
        
        Overview: Our diabetes management program provides comprehensive support
        for employees and dependents diagnosed with Type 1 or Type 2 diabetes.
        
        Included Services:
        - Continuous glucose monitoring (CGM) supplies covered at 90%
        - Diabetes education sessions: 10 per year, covered at 100%
        - Insulin pumps: Covered under durable medical equipment, 80% after deductible
        - Lab work: Quarterly HbA1c tests covered at 100%
        """,
        "source": "diabetes_program.txt"
    }
]
