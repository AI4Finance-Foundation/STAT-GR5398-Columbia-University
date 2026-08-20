# GR5398 26 Spring: FinRobot Equity Research AI Agent Track

## Assignment 1

### 0. Target

In this assignment, we would like you to run a tutorial of FinRobot ( `/source_code/agent_annual_report.ipynb`) to automatically generate an annual financial report, and basically learn what you will do in this semester.

+ You will learn basic methods of designing a **Multi-Agent System** (**MAS**)
+ You will learn to use Expert/Shadow/UserProxy structure to get a better result
+ You should generate no less than 5 financial reports using different stocks in a same industry field (better if they have different stock trends)
  + You should start with `NVDA`, `AMD`, `INTC`, `APPL`, `GOOGL`, and then you can try some other fields
+ After generating these 5 reports, write a basic analysis on their performance according to their historical performance and current market information, and publish your analysis report (not AI-generated reports) as a blog onto [medium](https://medium.com/)
+ All of your code files and financial reports should be uploaded into a new folder in `/submissions` named as `Assignment1_Name_UNI` (NOT a new branch!)

To find for more detailed informations, please refer to [FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) and specific notebook file [agent_annual_report.ipynb](https://github.com/AI4Finance-Foundation/FinRobot/blob/master/tutorials_advanced/agent_annual_report.ipynb).

Assignment 1 Financial Reports Submission Due Day: Feb 20, 2026

### 1. Multi-Agent System

A **Multi-Agent System (MAS)** is a computational framework in which multiple autonomous agents interact with each other to accomplish a task that would be difficult or inefficient for a single agent to complete alone. Each agent has a well-defined role, its own reasoning process, and the ability to communicate with other agents through structured messages.

![Building Your First Multi-Agent System: A Beginner's Guide ...](https://www.kdnuggets.com/wp-content/uploads/Building-Your-First-Multi-Agent-System-A-Beginners-Guide_2.png)

In practice, a MAS decomposes a complex problem into smaller, role-specific subtasks and assigns them to different agents. These agents may collaborate, verify each other’s outputs, or operate under a central controller to ensure stability and reliability. Compared with single-agent systems, multi-agent systems offer improved robustness, interpretability, and scalability, making them especially suitable for high-stakes domains such as finance, healthcare, and decision support systems.

### 2. Expert/Shadow/UserProxy Structure

The **Expert / Shadow / UserProxy structure** is a controlled multi-agent design pattern for building reliable and automated reasoning systems.

In this structure, the **Expert agent** is responsible for generating the primary analysis or decision, acting as the main problem solver. The **Shadow agent** operates independently to review or validate the Expert’s reasoning, helping to identify potential errors, biases, or missing considerations. The **UserProxy agent** serves as a controller that manages the interaction flow, monitors completion criteria, and determines when the task should be terminated.

By separating execution, verification, and control into distinct agents, this structure improves robustness, interpretability, and stability compared to single-agent systems. It is particularly suitable for high-stakes applications such as financial analysis, where correctness and controlled termination are critical.

### 3. Optimization (Optional)

You can add some useful parts or financial ratios to let report reader have a better understanding of this company's performance.


