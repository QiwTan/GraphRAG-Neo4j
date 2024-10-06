# main.py

import subprocess
import sys

def run_preprocess():
    print("Starting PDF preprocessing...")
    subprocess.run([sys.executable, "preprocess_pdfs.py"])
    print("PDF preprocessing completed.")

def run_upload():
    print("Starting data upload to Neo4j...")
    subprocess.run([sys.executable, "upload_to_neo4j.py"])
    print("Data upload to Neo4j completed.")

def run_interactive_qa():
    print("Starting interactive QA...")
    subprocess.run([sys.executable, "hybrid_qa.py"])

def main():
    print("Select an option:")
    print("1. Preprocess PDFs")
    print("2. Upload data to Neo4j")
    print("3. Start interactive QA")
    print("4. Run all steps")
    choice = input("Enter your choice: ")

    if choice == '1':
        run_preprocess()
    elif choice == '2':
        run_upload()
    elif choice == '3':
        run_interactive_qa()
    elif choice == '4':
        run_preprocess()
        run_upload()
        run_interactive_qa()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()