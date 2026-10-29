import os
import json
import logging
import time
import tarfile
import io
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# External Dependencies
import openai
import chromadb
import docker
from docker.errors import DockerException
from duckduckgo_search import DDGS
from tenacity import retry, wait_exponential, stop_after_attempt

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SuperAgent")

# ==========================================
# Phase 1: Security & Sandboxing (DevOps/Sec)
# ==========================================
class SecureCodeSandbox:
    """
    Executes AI-generated Python code in a highly restricted Docker container.
    Prevents the AI from destroying the host system or making unauthorized network calls.
    """
    def __init__(self):
        try:
            self.client = docker.from_env()
            # Ensure the base image exists
            self.image_name = "python:3.11-slim"
            logger.info(f"Ensuring Docker image {self.image_name} is available...")
            self.client.images.pull(self.image_name)
        except DockerException as e:
            logger.error("CRITICAL: Docker is not running. Code execution is disabled for safety.")
            raise RuntimeError("Docker is required for secure code execution.") from e

    def execute_python(self, code: str, timeout_seconds: int = 15) -> str:
        """Executes python code in an isolated container and returns stdout/stderr."""
        container = None
        try:
            # Create a securely configured container
            container = self.client.containers.run(
                self.image_name,
                command=["python", "-c", code],
                detach=True,
                network_disabled=True,     # No internet access for the executed code
                mem_limit="128m",          # Prevent out-of-memory attacks
                cpu_quota=50000,           # Limit to 50% of one CPU core
                user="1000:1000",          # Run as non-root
                security_opt=["no-new-privileges:true"],
                cap_drop=["ALL"]           # Drop all Linux capabilities
)
            
            # Wait for execution with timeout
            result = container.wait(timeout=timeout_seconds)
            logs = container.logs().decode("utf-8")
            
            if result["StatusCode"]!= 0:
                return f"Execution Failed (Code {result['StatusCode']}):\n{logs}"
            
            return logs if logs else "Execution completed successfully with no output."
            
        except Exception as e:
            return f"Sandbox Error: {str(e)}"
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

# ==========================================
# Phase 2: Memory & Knowledge Base (RAG)
# ==========================================
class KnowledgeBase:
    """Local Vector Database using ChromaDB and OpenAI Embeddings for approved information."""
    def __init__(self, persist_directory: str = "./agent_memory"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.openai_client = openai.Client()
        
        # Create or load collection
        self.collection = self.client.get_or_create_collection(
            name="approved_knowledge",
            metadata={"hnsw:space": "cosine"}
)
        logger.info("Knowledge Base initialized.")

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def _get_embedding(self, text: str) -> List[float]:
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
)
        return response.data[0].embedding

    def ingest_document(self, doc_id: str, text: str, metadata: Dict[str, Any] = None):
        """Adds approved information to the agent's memory."""
        embedding = self._get_embedding(text)
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}]
)
        logger.info(f"Ingested document: {doc_id}")

    def search(self, query: str, top_k: int = 3) -> str:
        """Searches the knowledge base for context."""
        embedding = self._get_embedding(query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
)
        
        if not results["documents"][0]:
            return "No relevant approved information found in the knowledge base."
            
        context = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            context.append(f"Source: {meta.get('title', 'Unknown')} - Content: {doc}")
            
        return "\n\n".join(context)

# ==========================================
# Phase 3: Tool Registry
# ==========================================
class ToolRegistry:
    """Manages the tools the Agent can use."""
    
    def __init__(self):
        self.sandbox = SecureCodeSandbox()
        self.knowledge_base = KnowledgeBase()
        self.ddgs = DDGS()
        
        # Ensure workspace exists for file I/O
        self.workspace = os.path.abspath("./agent_workspace")
        os.makedirs(self.workspace, exist_ok=True)

    def get_openai_tools_schema(self) -> List[Dict]:
        """Returns the JSON schema of tools for OpenAI's API."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": "Search the local vector database for approved, verified information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query."}
},
                        "required": ["query"]
}
}
},
            {
                "type": "function",
                "function": {
                    "name": "execute_python_code",
                    "description": "Run Python code in a secure, network-isolated sandbox to test algorithms, analyze data, or invent things. Print results to stdout.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Complete, valid Python code to execute."}
},
                        "required": ["code"]
}
}
},
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the open internet for current events or general knowledge.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query."}
},
                        "required": ["query"]
}
}
},
            {
                "type": "function",
                "function": {
                    "name": "save_artifact",
                    "description": "Save an invention, specification, or completed code to the workspace file system.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Name of the file (e.g., 'blueprint.md', 'app.py')."},
                            "content": {"type": "string", "description": "The complete content to save."}
},
                        "required": ["filename", "content"]
}
}
}
]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Routes the tool call to the actual implementation."""
        logger.info(f"Executing Tool: {tool_name}")
        
        try:
            if tool_name == "search_knowledge_base":
                return self.knowledge_base.search(arguments["query"])
                
            elif tool_name == "execute_python_code":
                return self.sandbox.execute_python(arguments["code"])
                
            elif tool_name == "web_search":
                results = self.ddgs.text(arguments["query"], max_results=3)
                if not results:
                    return "No results found."
                return "\n".join([f"Title: {r['title']}\nSnippet: {r['body']}" for r in results])
                
            elif tool_name == "save_artifact":
                # Path traversal mitigation
                safe_filename = os.path.basename(arguments["filename"])
                filepath = os.path.join(self.workspace, safe_filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(arguments["content"])
                return f"Successfully saved to {filepath}"
                
            else:
                return f"Error: Tool '{tool_name}' not found."
                
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}")
            return f"Error executing {tool_name}: {str(e)}"

# ==========================================
# Phase 4: The Agentic ReAct Orchestrator
# ==========================================
class SuperAgent:
    """The Autonomous System that manages state, goals, and the ReAct Loop."""
    
    SYSTEM_PROMPT = """You are an advanced Artificial Superintelligence framework.
Your purpose is to work autonomously toward complex goals provided by the user.

CORE BEHAVIORS:
1. PLANNING: Before acting, think step-by-step. Break the goal into sub-tasks.
2. VERIFICATION: Prioritize `search_knowledge_base` for verified info before relying on `web_search` or intrinsic knowledge.
3. INVENTING & CODING: If asked to invent or code, write the code, use `execute_python_code` to test it, observe the output, fix any errors, and iterate until it works.
4. ARTIFACTS: Once a task, invention, or code module is complete and verified, use `save_artifact` to save the final result.
5. AUTONOMY: You will be called in a loop. Continue taking actions using tools until the user's overarching goal is fully achieved. When finished, summarize the results for the user.

CONSTRAINTS:
- The Python sandbox has NO INTERNET ACCESS. Do not write code that requires requests/http.
- Always validate your assumptions. If code fails, read the error log and try again.
"""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = openai.Client()
        self.tools = ToolRegistry()
        self.messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

    @retry(wait=wait_exponential(multiplier=1, min=2, max=15), stop=stop_after_attempt(5))
    def _call_llm(self) -> Any:
        """Wrapper for LLM call with retry logic for API resilience."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools.get_openai_tools_schema(),
            tool_choice="auto",
            temperature=0.2 # Lower temperature for more logical/focused execution
)

    def run_goal(self, goal: str, max_iterations: int = 15):
        """Main Agentic Loop (ReAct paradigm)."""
        logger.info(f"New Goal Registered: {goal}")
        self.messages.append({"role": "user", "content": f"GOAL: {goal}\nPlease begin planning and execution."})
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"--- Agent Loop Iteration {iteration}/{max_iterations} ---")
            
            response = self._call_llm()
            message = response.choices[0].message
            
            # Append AI's message to history (required for tool tracking)
            self.messages.append(message)
            
            # If AI outputs text, print it (Reasoning/Summary)
            if message.content:
                print(f"\n[\033[94mAgent Thinking / Output\033[0m]: {message.content}")

            # If AI decided to use tools
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                        
                    print(f"[\033[93mAgent Action\033[0m]: Using tool '{tool_name}' with args: {args}")
                    
                    # Execute tool
                    result = self.tools.execute_tool(tool_name, args)
                    
                    # Provide tool result back to the LLM
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": str(result)
})
                    print(f"[\033[92mObservation\033[0m]: Tool returned {len(str(result))} characters.")
            else:
                # If no tool calls, the AI has concluded its task
                logger.info("Goal achieved or AI requires user feedback.")
                break
                
        if iteration == max_iterations:
            logger.warning("Max iterations reached. The agent was forced to stop to prevent infinite loops.")


# ==========================================
# Phase 5: Execution / Entry Point
# ==========================================
if __name__ == "__main__":
    # Ensure API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        exit(1)

    print("Initializing SuperAgent System...")
    agent = SuperAgent(model="gpt-4o") # You can switch to gpt-4-turbo if preferred
    
    # 1. OPTIONAL: Seed the Knowledge base with "Approved Information"
    print("\n--- Phase 1: Ingesting Approved Knowledge ---")
    kb = agent.tools.knowledge_base
    kb.ingest_document(
        doc_id="doc_1",
        text="The company standard for sorting algorithms is strictly Merge Sort due to stable time complexity requirements. Do not use Bubble Sort or Quick Sort.",
        metadata={"title": "Engineering Standards v1"}
)
    kb.ingest_document(
        doc_id="doc_2",
        text="Project X requires all configuration files to be output as valid JSON, not YAML.",
        metadata={"title": "Project X Requirements"}
)
    print("Knowledge base updated.")

    # 2. RUN THE AGENTIC LOOP
    print("\n--- Phase 2: Autonomous Execution ---")
    # Complex Goal: Test RAG, test Code Execution, test iteration, test File I/O
    user_goal = """
    1. Search the knowledge base for our company's preferred sorting algorithm.
    2. Write a Python function implementing that specific algorithm.
    3. Generate a list of 50 random integers and use the algorithm to sort them.
    4. Execute the code to verify it works perfectly.
    5. Once verified, save the working python code to a file called 'company_sort.py'.
    """
    
    agent.run_goal(user_goal)
    
    print("\nSystem execution complete. Check the './agent_workspace' directory for saved artifacts.")
