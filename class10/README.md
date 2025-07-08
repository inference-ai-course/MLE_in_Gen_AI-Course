# 🔗 Model Context Protocol (MCP) Tutorial

A comprehensive tutorial and implementation of the Model Context Protocol (MCP) for building AI agents that can seamlessly interact with external tools and data sources.

## 🎯 Overview

This project contains two main components that work together to demonstrate modern AI agent development:

1. **MCP Tutorial** (`MCP_Tutorial.ipynb`) - A comprehensive guide to building Model Context Protocol (MCP) systems that enable Large Language Models (LLMs) to interact with external tools, APIs, and data sources through a standardized protocol.

2. **LangGraph Agent Examples** (`class_10_code.ipynb`) - Practical implementations of LangGraph agents for building AI applications with complex workflows, state management, and multi-step decision making.

### How They Work Together

- **MCP** provides the foundation for tool integration and external data access
- **LangGraph** provides the framework for orchestrating complex AI workflows
- **Combined** they enable powerful AI agents that can access tools, maintain state, and execute sophisticated multi-step processes

The MCP tutorial teaches you how to create modular, interoperable tools, while the LangGraph examples show you how to build intelligent agents that can use those tools in complex workflows.

## 🤔 What is MCP?

Model Context Protocol (MCP) is an open standard that creates a bridge between AI models and the broader digital ecosystem. It enables:

- **Standardized Communication**: Universal protocol for LLM-tool interaction
- **Dynamic Context Updates**: Real-time information during conversations
- **Modular Architecture**: Easy to add/remove capabilities
- **Interoperability**: Works across different LLM applications

### MCP vs Traditional Function Calls

| Aspect                     | Traditional Function Calls | MCP                    |
| -------------------------- | -------------------------- | ---------------------- |
| **Protocol**         | Custom implementation      | Standardized protocol  |
| **Flexibility**      | Hard-coded functions       | Dynamic tool discovery |
| **Scalability**      | Limited                    | Highly scalable        |
| **Interoperability** | App-specific               | Cross-platform         |
| **Maintenance**      | High overhead              | Low maintenance        |

## 🕸️ What is LangGraph?

LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends the LangChain ecosystem by providing a framework for creating complex AI workflows with:

- **State Management**: Persistent memory and context across interactions
- **Multi-Agent Systems**: Coordination between multiple AI agents
- **Workflow Orchestration**: Complex decision trees and branching logic
- **Tool Integration**: Seamless connection with external APIs and data sources
- **Human-in-the-Loop**: Interactive workflows with human oversight

### LangGraph vs Traditional Chatbots

| Aspect                | Traditional Chatbots | LangGraph Agents      |
| --------------------- | -------------------- | --------------------- |
| **Memory**            | Limited context      | Persistent state      |
| **Complexity**        | Linear conversations | Multi-step workflows  |
| **Decision Making**   | Simple responses     | Conditional logic     |
| **Tool Usage**        | Basic integrations   | Advanced orchestration|
| **Scalability**       | Single agent         | Multi-agent systems   |

### Key LangGraph Concepts

- **Nodes**: Individual processing units in the workflow
- **Edges**: Connections that define the flow between nodes
- **State**: Persistent data that flows through the graph
- **Checkpoints**: Save/restore points for long-running workflows
- **Human-in-the-Loop**: Integration points for human decision making

## 📁 Project Structure

```
class10/
├── MCP_Tutorial.ipynb              # Main MCP tutorial and implementation
├── class_10_code.ipynb             # LangGraph agent examples and workflows
├── mcp_sandbox/                    # File system sandbox for MCP operations
│   ├── mcp_example.txt
│   ├── calculation.txt
│   ├── research_summary.txt
│   └── research_artificial_intelligence_in_healthcare.txt
├── chroma_db/                      # Vector database for embeddings
├── Class_10_presentation.pdf       # Presentation slides
├── Class_10_presentation.pptx      # Presentation slides
├── U.S._Economic_Outlook.pdf       # Sample data for LangGraph agents
├── Stock_Market_Performance_2024.pdf # Sample data for LangGraph agents
└── README.md                       # This file
```

## ✨ Features

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Dependencies

```bash
# for Langgraph
pip install langchain langchain-openai langchain-community langgraph chromadb pypdf python-dotenv langchain_chroma

# for MCP
pip install requests aiohttp asyncio fastapi uvicorn httpx
```

### 🚀 Quick Start

### 1. MCP Tutorial

Open `MCP_Tutorial.ipynb` to learn about Model Context Protocol:

- **Simple MCP Server**: Basic computational tools (calculator, text processor)
- **File System MCP**: Safe file operations in sandbox environment
- **Web Search MCP**: External data retrieval capabilities
- **MCP Client**: Multi-server orchestration and workflow execution
- **Advanced Patterns**: Stateful servers, composite tools, production features

### 2. LangGraph Agent Examples

Open `class_10_code.ipynb` to explore LangGraph agent implementations:

- **Agent Workflows**: Complex multi-step AI processes with conditional logic
- **State Management**: Persistent conversation and context handling across sessions
- **Tool Integration**: Connecting agents with external APIs, databases, and MCP servers
- **PDF Processing**: Document analysis and information extraction from sample documents
- **Multi-Agent Coordination**: Systems where multiple agents work together
- **Human-in-the-Loop**: Interactive workflows that require human decision making

**For more detailed LangGraph learning**, check out this comprehensive repository: https://github.com/ScottLL/langgraph_lib

### 3. Sample Data

The project includes sample PDF documents for testing:

- `U.S._Economic_Outlook.pdf` - Economic analysis data
- `Stock_Market_Performance_2024.pdf` - Financial market data

These documents can be used with the LangGraph agents for document processing and analysis.

## 📖 How to Use the Files

### MCP Tutorial (`MCP_Tutorial.ipynb`)

This notebook contains the complete MCP implementation:

1. **Run all cells** to set up the environment and import libraries
2. **Follow the sections** in order to understand MCP concepts
3. **Execute the examples** to see MCP servers in action
4. **Test the workflows** to understand multi-server orchestration
5. **Explore advanced patterns** for production-ready implementations

The tutorial creates a `mcp_sandbox/` directory for safe file operations during learning.

### LangGraph Agent Examples (`class_10_code.ipynb`)

This notebook demonstrates advanced LangGraph agent patterns:

1. **Set up the environment** with LangGraph and related dependencies
2. **Load and process sample documents** from the included PDF files using vector databases
3. **Build and run agent workflows** to see complex multi-step AI processes in action
4. **Experiment with state management** and persistent conversation flows across sessions
5. **Test tool integration** with external APIs, databases, and MCP servers
6. **Explore multi-agent coordination** and human-in-the-loop workflows

### Sample Documents

- **U.S._Economic_Outlook.pdf**: Use with economic analysis agents
- **Stock_Market_Performance_2024.pdf**: Use with financial analysis agents

These documents provide real-world data for testing LangGraph agent capabilities.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive tests for new features
- Update documentation for API changes
- Keep functions under 50 lines
- Add proper error handling

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Model Context Protocol specification and community
- Anthropic for MCP development
- Open source contributors and maintainers

## 📞 Support

For questions, issues, or contributions:

1. **MCP Questions**: Check the [MCP Tutorial notebook](MCP_Tutorial.ipynb) for detailed examples and explanations
2. **LangGraph Questions**: Review the [LangGraph agent examples](class_10_code.ipynb) for workflow patterns
3. **General Issues**: Open an issue on the repository
4. **Community**: Join the MCP and LangGraph community discussions

## 🎯 Learning Path

1. **Start with MCP**: Begin with `MCP_Tutorial.ipynb` to understand tool integration and external data access
2. **Explore LangGraph**: Move to `class_10_code.ipynb` for advanced agent patterns and workflow orchestration
3. **Combine Both**: Use MCP servers as tools within LangGraph agents for powerful AI systems
4. **Build Your Own**: Create custom agents and MCP servers for your specific use cases
5. **Advanced Patterns**: Explore multi-agent systems, human-in-the-loop workflows, and production deployments

### Use Cases

- **Document Analysis**: Process PDFs and extract insights using LangGraph workflows
- **Research Assistants**: Combine MCP tools with LangGraph state management
- **Multi-Agent Systems**: Coordinate multiple specialized agents
- **Interactive Applications**: Build human-in-the-loop AI systems

---

**🎉 Happy AI Agent Development!**

This project provides a comprehensive foundation for building AI agents that can seamlessly interact with external tools and services through both MCP and LangGraph patterns.
