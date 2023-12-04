import os
from datetime import datetime
import json
# ....................................................................................................
def generate_readme(data):
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

![Figure 1: Example figure](fig.png)
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
- Email: [Email]{data['instructor_email']}
- GitHub: [GitHub Repo]({data['github_repo']})
"""
    # .......................................................
    with open(f"proposal_{data['instructor']}.md", "w") as readme_file:
        readme_file.write(readme_template)


# -------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    input_file_path = 'input.json'
    with open(input_file_path, 'r') as input_file:
        json_data = json.load(input_file)
    generate_readme(json_data)
    print(f"proposal_{json_data['instructor']}.md generated successfully.")
