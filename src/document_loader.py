from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader

def load_documents(data_dir: str = "data/raw") -> List:
    data_path = Path(data_dir)
    documents = []
    for file_path in data_path.rglob("*"):
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            documents.extend(docs)
        elif file_path.suffix.lower() in [".docx", ".doc"]:
            loader = UnstructuredWordDocumentLoader(str(file_path))
            docs = loader.load()
            documents.extend(docs)
    return documents

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
