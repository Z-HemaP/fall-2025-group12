
## Step 1: GitHub Rulesets
- Go to your repository
- Click on setting
- On the left pane find Rules
- Click on Rules and expend it and find rulesets


## Step 2: Repository Ruleset

**Ruleset Name** 

- main 
![1.png](figs/1.png)
**Enforcement status** 


- Active

**Bypass List:**  

- Add bypass on the right side
- Check bix the Repository Admin Role
- It makes a bypass list to Always allow

**Target Branches:**  

- Include: Default Branch

**Enforcement Status:** 

- Active  


**Rules**
Check box the below items:
![2.png](figs/2.png)
1. **Restrict Deletions**
![3.png](figs/3.png)
2. **Require a Pull Request Before Merging**
 - **Required Approvals:** 1.
 - **Dismiss Stale Approvals:** Enabled (approvals are dismissed when new commits are pushed).
![4.png](figs/4.png)
3. **Block Force Pushes** 

