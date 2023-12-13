import re
import pdfplumber

pdf_path = "test.pdf"

questions = []
answers = []

with pdfplumber.open(pdf_path) as pdf:
    pages = pdf.pages
    for page in pages:
        text = page.extract_text()

        # Remove everything before the first question
        text = re.sub(r".*(?=1\. What is your name\?)", "", text, flags=re.DOTALL)
        text = re.sub(r"e.g., LAC Transportation or Energy Global Practice", "", text)
        text = re.sub(r"i.e., which county, city, country, and/or region", "", text)

        text = re.sub(r"e.g., early ideation; concept note; supervision; etc.", "", text)
        text = re.sub(r"STATUS AND TIMELINE", "", text)
        text = re.sub(r"e.g., 100 GB", "", text)
        text = re.sub(r"CHALLENGE", "", text)
        text = re.sub(r"AM UNIVERSITY DATA FELLOWS: PROPOSAL SUBMISSION FORM", "", text)
        text = re.sub(r"e.g., If the project requires student access to data not owned by the World Bank, students may need to be hired as non-fee STCs.", "", text)
        text =re.sub(
            r"Could the students immediately access the data upon project commencement\? If your data requires additional World Bank\ncredentials to gain access, could you commit to helping the student gain the necessary credentials within 10-days of the start of\nthe project\? Are there other uncertainties around data access that the program should be aware of\?",
            "",
            text
        )

        text = re.sub(r"Could the students immediately access+that the program should be aware of?", "", text)
        text = re.sub(r"https://forms\.office\.com[^\s]*", "", text)

        text = re.sub(r"1/3*", "", text)
        text = re.sub(r"2/3*", "", text)
        text = re.sub(r"3/3*", "", text)


        # Extract questions and answers
        answer_regex = re.compile(r"(\b[A-Z]\S.*)", re.DOTALL)


        for match in re.finditer(answer_regex, text):
            answers.append(match.group(1))



with open("answers.txt", "w") as f:
    for answer in answers:
        f.write(answer + "\n")

import re
import json

qa_dict = {}

with open("answers.txt") as f:
    text = f.read()

    # Get all lines with ? indicating a question
    questions = re.findall(r'^.+[\?\*]', text, re.M)

    # Use a pattern that includes multiple lines for answers
    answers = re.split(r'^.+[\?\*]', text, flags=re.M)[1:]

    for i, ques in enumerate(questions):
        # Combine the multiline answer until the next question
        answer = re.search(r'^.*?(?=\n\s*\n*\d+\.)', answers[i], re.DOTALL)
        qa_dict["Q" + str(i + 1)] = ques.strip()
        qa_dict["A" + str(i + 1)] = answer.group(0).strip() if answer else answers[i].strip()

    # Extract question and answer for question 15
    match = re.search(r'15\.(.*?)(?=\d+\.)', text, re.DOTALL)
    if match:
        qa_dict["Q15"] = match.group(1).strip()

        # Find the corresponding answer
        start_index = match.end(1)  # Use end position of the matched group
        end_index = text.find("\n\n", start_index)
        qa_dict["A15"] = text[start_index:end_index].strip()

with open("qa.json", "w") as f:
    json.dump(qa_dict, f, indent=4)
