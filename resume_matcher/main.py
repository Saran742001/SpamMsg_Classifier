import sys
import os

# Add the project directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse

from src.load_data import load_job_description, load_resumes
from src.preprocessing import clean_text
from src.vectorizer import vectorize_texts
from src.similarity import calculate_similarity
from src.keywords import extract_keywords


def parse_args():
    parser = argparse.ArgumentParser(
        description="📄 Resume Matcher using NLP"
    )

    parser.add_argument(
        "--job",
        type=str,
        help="Path to job description file (optional)"
    )

    parser.add_argument(
        "--resumes",
        type=str,
        required=True,
        help="Folder containing resume .txt files"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top resumes to display (default: 3)"
    )

    parser.add_argument(
        "--min_score",
        type=float,
        default=0.0,
        help="Minimum similarity score to include (default: 0.0)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n📄 Resume Matcher\n")

    # Load job description
    if args.job:
        job_desc = load_job_description(args.job)
    else:
        # load_job_description handles the interactive input loop if path is None
        job_desc = load_job_description()

    job_desc = clean_text(job_desc)

    # Load resumes
    resumes = load_resumes(args.resumes)
    resumes = {k: clean_text(v) for k, v in resumes.items()}

    print(f"✅ Loaded {len(resumes)} resumes")

    # Vectorization
    tfidf_matrix, vectorizer = vectorize_texts(job_desc, resumes)

    # Similarity
    scores = calculate_similarity(tfidf_matrix, resumes)

    # Keywords
    # Pass tfidf_matrix (specifically the job vector part is handled in the function)
    job_keywords = extract_keywords(vectorizer, tfidf_matrix)

    # Rank resumes
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n🔢 Showing top {args.top_k} resumes (min_score={args.min_score})\n")

    count = 0
    for resume, score in ranked:
        if score < args.min_score:
            continue

        resume_words = set(resumes[resume].split())
        matched = job_keywords.intersection(resume_words)

        count += 1
        print(f"{count}. {resume}")
        print(f"   🔹 Match Score: {score:.4f}")
        print(f"   🔹 Matched Keywords: {', '.join(matched) if matched else 'None'}\n")

        if count == args.top_k:
            break

    if count == 0:
        print("❌ No resumes matched the criteria.")


if __name__ == "__main__":
    main()
