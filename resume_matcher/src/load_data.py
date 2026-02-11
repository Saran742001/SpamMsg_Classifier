import os
import pdfplumber


def load_job_description(job_path=None):
    if job_path:
        with open(job_path, "r", encoding="utf-8") as f:
            return f.read()

    print("📝 Enter job description (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)

    return " ".join(lines)


def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
    return text


def load_resumes(folder_path):
    resumes = {}

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                resumes[file] = f.read()

        elif file.endswith(".pdf"):
            resumes[file] = extract_text_from_pdf(file_path)

    if not resumes:
        raise ValueError("❌ No resume files found (.txt or .pdf)")

    return resumes
