# LLM-Value-Analyzer
A data analysis project to help businesses choose the right Large Language Model (LLM) for their needs and budget.

Demo: https://llm-value-analyzer-fc4vjkxd7qzet5ztw2otha.streamlit.app/
### 1. The Problem
Companies are spending millions on AI but often choose models based on hype rather than data. Choosing the wrong model means wasting money on overpowered tools or getting poor results from underpowered ones.
My goal was to answer the key business question: "For any given task, which LLM provides the best balance of performance, speed, and cost?"

### 2. The Process
To solve this, I acted as a freelance data consultant.
#### Data: 
I used the "Large Language Models Comparison Dataset," which includes metrics for 200 models covering their performance (Benchmark (MMLU)), speed (tokens/sec), and cost (Price / Million Tokens).

#### Feature Engineering: 
The raw data doesn't show "value." To find it, I created two critical new metrics:
-> performance_per_dollar: This shows how much "smartness" (MMLU score) you get for every $1 spent.
-> speed_per_dollar: This shows how much "speed" (tokens/sec) you get for every $1 spent.

#### Visualization: 
I used Plotly to create 11 interactive charts to find the winners, losers, and strategic insights.

### 3. Key Features & Visualizations
The dashboard is built with Streamlit and Plotly, allowing for fully interactive filtering.

##### 1. The Value Quadrant
This is the main chart of the analysis. It plots every model on a 2x2 grid to instantly identify which models are "Winners" (High Performance, Low Cost) and which are "Traps" (Low Performance, High Cost).

<img width="1087" height="334" alt="image" src="https://github.com/user-attachments/assets/024f461c-4921-4868-926c-42b3a901b2a6" />

##### 2. "Value-for-Money" Rankings
I engineered two new features to find the true value of each model:
-> performance_per_dollar: Shows which models give the most "smartness" (MMLU benchmark score) for every $1 spent.
-> speed_per_dollar: Shows which models give the most "speed" (tokens/sec) for every $1 spent.

<img width="1053" height="318" alt="image" src="https://github.com/user-attachments/assets/4e304b54-1c12-43d4-8c58-45ccb234bd81" />

<img width="1057" height="316" alt="image" src="https://github.com/user-attachments/assets/e8b3fd72-0414-4ad0-85a7-07e4f3f4f252" />

##### 3. "Buy vs. Build" Analysis
A high-level strategic chart that compares the average performance_per_dollar of Open-Source models versus Proprietary (closed-source) models.
<img width="1089" height="327" alt="image" src="https://github.com/user-attachments/assets/d016d039-9d9a-44ce-81bf-bea90da790d5" />

##### 4. Interactive Filtering
The sidebar allows any user to filter the entire dashboard by:
-> Provider (e.g., OpenAI, Google, Meta)
-> Model Type (Open-Source vs. Proprietary)
-> Simple Ratings (Quality, Speed, Price)
-> Technical Specs (Context Window, Training Data Size, Energy Efficiency)

#### How to Run This Project Locally

##### 1.Clone this repository:
git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/llm-dashboard-project.git
cd llm-dashboard-project

##### 2.Install the required libraries:
Make sure you have llm_dashboard.py, llm_comparison_dataset.csv, and requirements.txt in the same folder.
pip install -r requirements.txt

##### 3.Run the Streamlit app:
streamlit run llm_dashboard.py


