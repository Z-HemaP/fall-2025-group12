import os
from datetime import datetime
import json
import shutil
# ....................................................................................................
def generate_readme(data, output_file_path):
    readme_template = \
f"""
# Capstone Proposal
## {data['project_name']}
### Proposed by: {data['Proposed by']}
#### Email: {data['Proposed by email']}
#### Advisor: {data['instructor']}
#### The George Washington University, Washington DC  
#### Data Science Program


## 1 Objective:  
{data['Objective']}

![Figure 1: Example figure]({data['Year']}_{data['Semester']}_{data['Version']}.png)
*Figure 1: Caption*

## 2 Dataset:  
{data['Dataset']}

## 3 Rationale:  
{data['Rationale']}

## 4 Approach:  
{data['Approach']}

## 5 Timeline:  
{data['Timeline']}

## 6 Expected Number Students:  
{data['Expected Number Students']}

## 7 Possible Issues:  
{data['Possible Issues']}


## Contact
- Author: {data['instructor']}
- Email: [{data['instructor_email']}](Eamil)
- GitHub: [{data['github_repo']}](Git Hub rep)
"""
    # .......................................................
    with open(output_file_path + f"proposal_{data['Version']}.md", "w") as readme_file:
        readme_file.write(readme_template)



# -------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    input_file_path = 'input.json'
    Year = "2024"
    Semester = "Fall"
    Version = "2"


    output_file_path = os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{Year}{os.sep}{Semester}{os.sep}{Version}{os.sep}'
    with open(output_file_path+ input_file_path, 'r') as input_file:
        json_data = json.load(input_file)
    generate_readme(json_data, output_file_path)
    try:
        shutil.copy(f"{json_data['Year']}_{json_data['Semester']}_{json_data['Version']}.png",output_file_path )
    except:
        print('No Figure is needed.')
    print(f"proposal_{json_data['instructor']}.md generated successfully.")
