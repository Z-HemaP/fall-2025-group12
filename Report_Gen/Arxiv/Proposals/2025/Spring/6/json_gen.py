import json
import os
import shutil


def save_to_json(data, output_file_path):
    with open(output_file_path, 'w') as output_file:
        json.dump(data, output_file, indent=2)


data_to_save = \
    {
        # -----------------------------------------------------------------------------------------------------------------------
        "Version":
            """6""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Year":
            """2025""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Semester":
            """Spring""",
        # -----------------------------------------------------------------------------------------------------------------------
        "project_name":
            """Increasing Intellectual Humility with genAI Chatbots""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """ 
            The goal of this project is to design, develop, and validate a generative AI chatbot that promotes intellectual humility in students through 
            structured dialogue and debate. Specifically, the project will focus on:
            Customizing an open-source LLM (e.g., LLaMA3) using Retrieval-Augmented Generation (RAG) and/or fine-tuning techniques (e.g., LoRA) to create a
            chatbot tailored for fostering intellectual humility.
            Evaluating the chatbot's performance (e.g., adherence to instructions, depth of argumentation)
            by simulating diverse student interactions using multiple LLM agents.
            Conducting user interface (UI) and usability testing to improve user experience.
            Optionally, integrating Learning Tools Interoperability (LTI) features to enable seamless usage in Learning Management Systems (LMS)
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            The dataset for fine-tuning and customization is still TBD. However, data will likely consist of relevant open-source academic texts,
             dialogue datasets, and domain-specific content to build the required context for structured conversations.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            Generative AI (genAI) technology offers a novel approach to supporting social science research. By using LLM-based chatbots for educational
             interventions, there is potential to improve not only how learning experiences are delivered but also to enhance the efficiency and 
             scalability of assessing student learning outcomes. In this project, we will explore how an AI-driven conversation can both act as an educational 
             intervention and provide meaningful metrics for learner assessment.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            The project will proceed in several phases:
            **Requirement Analysis**: Collaborate with lead researchers to define key components of intellectual humility and specify chatbot performance
            requirements.
            **Development**: Customize an open-source LLM to create the initial version of the chatbot using RAG and fine-tuning methods.
            **Validation**: Conduct multi-agent testing to evaluate the chatbot's performance across multiple domains and with varied inputs, focusing on 
            metrics such as adherence to debate guidelines, argument quality, and ability to stay on topic.
            **Usability Evaluation**: Perform usability testing with end users, including metrics for user satisfaction, ease of use, and learning outcomes.
            **(Optional) Integration**: Develop LTI integration to connect the chatbot with existing LMS tools to ensure accessibility for students in course
            environments.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            This is a rough timeline for the project:
             **Weeks 1-2**: Familiarization with project requirements, intellectual humility constructs, and open-source tools.
             **Weeks 3-8**: Design, develop, and build a working prototype.
             **Weeks 9-12**: Iterative testing and improvement of chatbot performance, incorporating feedback from simulated interactions.
             **Weeks 13-16**: Multi-agent testing and usability evaluation.
             **Weeks 17-18**: Final reporting, documentation, and presentation of outcomes.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            This project is suitable for a team of 2-3 students, given the scope and need for interdisciplinary collaboration.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            Potential challenges for this project include:
             **Limited Training Data**: Finding appropriate datasets for Retrieval-Augmented Generation or fine-tuning the LLM may be a limiting factor.
             **Complexity of Teaching Intellectual Humility**: Designing prompts and interactions to genuinely foster intellectual humility can be difficult
             , requiring careful iteration and feedback.
             **Interdisciplinary Collaboration**: This project spans multiple disciplines (e.g., Education, Psychology, Data Science, Software Engineering), 
             requiring effective collaboration and understanding across fields.
            """,

        # -----------------------------------------------------------------------------------------------------------------------
        "Proposed by": "Dr. Ryan Watkins, Phil Moore, and Nils Olsen",
        "Proposed by email": "rwatkins@gwu.edu",
        "instructor": "Amir Jafari",
        "instructor_email": "ajafari@gmail.com",
        "github_repo": "https://github.com/amir-jafari/Capstone",
        # -----------------------------------------------------------------------------------------------------------------------
    }
os.makedirs(
    os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}',
    exist_ok=True)
output_file_path = os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}{os.sep}'
save_to_json(data_to_save, output_file_path + "input.json")
shutil.copy('json_gen.py', output_file_path)
print(f"Data saved to {output_file_path}")
