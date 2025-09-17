# Lang-graph-project

Lang-graph-project is a comprehensive repository designed to explore, implement, and demonstrate concepts related to LangGraph - a library for building stateful, multi-actor applications with large language models (LLMs). This project contains practical examples, tutorials, and implementations covering various aspects of agent-based AI systems.

## Features

- **LangGraph Fundamentals**: Basic concepts and state management
- **Tool Calling**: Integration with external APIs and tools
- **Human-in-the-Loop (HITL)**: Interactive workflows requiring human input
- **Multi-Agent Architecture**: Collaborative agent systems
- **Plan and Execute**: Strategic planning and execution patterns
- **Agentic RAG**: Retrieval-Augmented Generation with agent capabilities
- **Memory Management**: Persistent state across conversations
- **ReAct Patterns**: Reasoning and Acting frameworks

## Getting Started

### Prerequisites

- Python 3.8 or higher recommended
- [pip](https://pip.pypa.io/en/stable/) for dependency management
- API keys for OpenAI, Azure, Anthropic, or Gemini (depending on usage)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kardwalker/Lang-graph-project.git
   cd Lang-graph-project
   ```

2. Create a virtual environment and install dependencies:

   **Mac/Linux/WSL:**
   ```bash
   python3 -m venv langgraph_env
   source langgraph_env/bin/activate
   pip install -r requirements.txt
   ```

   **Windows:**
   ```powershell
   python3 -m venv langgraph_env
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   langgraph_env\Scripts\activate
   pip install -r requirements.txt
   ```

### Usage

Explore the example scripts in the `langgraph_env/` directory:

```bash
# Basic LangGraph concepts
python langgraph_env/lesson1.py

# Tool calling examples
python langgraph_env/basic_tool_calling_01.py

# Human-in-the-loop examples
python langgraph_env/HITL/basic.bp_01.py

# Multi-agent examples
python langgraph_env/Multi_Agent_architecture/multi_agent_start.py
```

## Project Structure

```
Lang-graph-project/
├── README.md
├── requirements.txt
├── LICENSE
├── langgraph_env/              # Main code directory
│   ├── lesson1.py             # Basic LangGraph introduction
│   ├── basic_tool_calling_01.py
│   ├── agent_with_memory.py
│   ├── HITL/                  # Human-in-the-Loop examples
│   ├── Multi_Agent_architecture/
│   ├── Plan_and_Execute/      # Planning and execution patterns
│   ├── Agentic_RAG/          # RAG with agents
│   ├── PROJECT/              # Main project implementations
│   ├── ReAct/                # ReAct pattern examples
│   └── Testing/              # Test files
├── simplified_rag_workflow.png
├── graph_309.png
├── graph_9869.png
└── 2311.12983v1.pdf          # Research paper reference
```

## API Configuration

Set your API keys in your environment:

```bash
export OPENAI_API_KEY="your-openai-key"
export AZURE_OPENAI_API_KEY="your-azure-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-gemini-key"
```

Or create a `.env` file in the project root with your keys.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements or new features.

1. Fork this repository
2. Create your feature branch (`git checkout -b my-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin my-feature`)
5. Open a pull request

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Contact

For questions or suggestions, open an issue or contact the maintainer via GitHub.