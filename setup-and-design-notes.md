# Super Agent — setup and design notes
Transcribed from Hattie's screenshots (the "New A.i" app) sent 2026-10-25. Original screenshots are in the screenshots/ subfolder.

## Intro (screenshot 2)
To build an advanced, goal-oriented AI assistant (a foundational step toward an Artificial Superintelligence architecture), we must construct a highly capable Agentic Framework.

This system requires several decoupled but integrated components:
1. Core Reasoning Engine: An advanced LLM (we will use OpenAI's GPT-4o) capable of complex logic, planning, and function calling.
2. Episodic & Semantic Memory (RAG): A vector database (ChromaDB) to ingest, embed, and retrieve "approved information."
3. Secure Execution Environment: A sandboxed Docker environment to write and test code securely without compromising the host machine.
4. Tool Registry: Web search, file system operations, and memory querying.
5. Agentic Loop: A ReAct (Reason, Act, Observe) loop that allows the AI to autonomously plan steps, execute tools, evaluate output, and iterate until the goal is achieved.

Below is the complete, production-ready, highly secure Python implementation.

## Prerequisites & Setup (screenshot 1)
1. Install Docker: You must have Docker installed and running on your system. The system uses Docker to create a secure, network-isolated, resource-constrained sandbox for the AI to execute the code it writes.
2. Environment Variables: Set your OpenAI API key.
   export OPENAI_API_KEY="your-api-key-here"
3. Install Dependencies: Create a requirements.txt file (see requirements.txt in this folder), then:
   pip install -r requirements.txt

The System Implementation (super_agent.py) — see super_agent.py in this folder.

## Design notes (screenshots 3-4)
1. Robust Security (SecOps):
- LLMs can hallucinate malicious code. Executing subprocess.run natively is a massive security vulnerability.
- I implemented SecureCodeSandbox using the docker API. It runs a lightweight python container with network_disabled=True (prevents downloading malware/data exfiltration), strict memory limits (128m), drops root privileges, and drops all kernel capabilities.

2. True RAG (Data Architecture):
- No mock implementations. I utilized ChromaDB (a persistent, local vector database) and OpenAI's text-embedding-3-small.
- The agent is explicitly prompted to search this database first before making decisions.

3. Resilience & Reliability (QA / DevOps):
- The tenacity library is used to wrap API calls with Exponential Backoff retries. If the OpenAI API rate limits or drops the connection, the system gracefully waits and retries without crashing the agent loop.
- Robust try/catch blocks around JSON parsing for OpenAI tool arguments, handling cases where the LLM outputs malformed JSON.

4. Agentic ReAct Loop (Software Engineering):
- Instead of rigidly tying tools to a linear script, the run_goal loop mimics a cognitive architecture. The LLM reasons, decides on a tool (Action), the tool executes (Observation), and the LLM receives the output to decide its next move until the overarching goal is satisfied.

5. Separation of Concerns:
- KnowledgeBase (Storage) is isolated from SecureCodeSandbox (Compute), which is isolated from ToolRegistry (Routing), which is isolated from SuperAgent (Brain). This adheres to strict SOLID principles making it easy to swap OpenAI for Anthropic Claude later.

## Quota screenshot (screenshot 5)
The "New A.i" app showed: "You've used your free messages quota. Your quota resets tomorrow, Oct 25, 2026 at 7:14 AM." — sent by Hattie 2026-10-25 2:56 AM CDT.
