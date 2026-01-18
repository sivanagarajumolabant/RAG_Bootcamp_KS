# 🚀 Text-to-SQL with LlamaIndex

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.14+-10A37F?style=for-the-badge)](https://www.llamaindex.ai/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**A progressive series of Jupyter notebooks demonstrating text-to-SQL capabilities using LlamaIndex** 📚

*From basic queries to advanced workflows with cloud database integration* ✨

</div>

---

## 📖 Overview

This project provides a **hands-on learning path** for building text-to-SQL systems using LlamaIndex. You'll progress from basic SQLite queries to advanced workflows with vector-based table retrieval and production PostgreSQL/Supabase integration.

🎯 **Perfect for:** Data engineers, ML engineers, and developers looking to integrate natural language querying into their applications.

## ✅ Prerequisites

- 🐍 Python 3.12+
- 🔑 OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- 📊 Basic understanding of SQL and Python
- 📓 Jupyter Notebook or JupyterLab

## ⚡ Quick Start

### 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd text2sql-rag
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv

# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
```

Your `.env` file should contain:
```bash
OPENAI_API_KEY=sk-proj-your-actual-key-here

# Optional: For Notebook 04 (PostgreSQL/Supabase)
POSTGRES_HOST=your_supabase_host
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=postgres
POSTGRES_PORT=5432
```

### 5️⃣ Start Jupyter

```bash
jupyter notebook
```

## 📚 Notebooks

### 📘 01 - Basic Text-to-SQL
**Level:** 🟢 Beginner | **Duration:** ⏱️ 30-45 minutes

**Topics:**
- 🎯 Introduction to text-to-SQL concepts
- 🔧 Using `NLSQLTableQueryEngine` for simple queries
- 💾 Basic SQLite operations
- 👥 Working with employee/department sample data
- 🔍 Understanding generated SQL
- 🔒 Security considerations

**💡 What you'll learn:** How to create a basic natural language to SQL query interface

---

### 📗 02 - Intermediate Text-to-SQL
**Level:** 🟡 Intermediate | **Duration:** ⏱️ 45-60 minutes

**Topics:**
- 🗂️ Multi-table database queries
- 🎯 `SQLTableRetrieverQueryEngine` with ObjectIndex
- 🔄 Dynamic table retrieval for large schemas
- 🧠 Vector-based table schema matching
- 🛒 E-commerce analytics use case
- 🦆 DuckDB integration (bonus section)

**💡 What you'll learn:** How to build systems that automatically find relevant tables for complex queries

---

### 📙 03 - Advanced Text-to-SQL
**Level:** 🔴 Advanced | **Duration:** ⏱️ 60-90 minutes

**Topics:**
- ⚙️ LlamaIndex Workflows architecture
- 🔍 Query-time table retrieval
- 📊 Query-time row retrieval using vector indices
- 📚 WikiTableQuestions dataset
- 🛡️ Production error handling
- ✅ SQL query validation
- 🔄 Event-driven workflows

**💡 What you'll learn:** How to build production-ready text-to-SQL systems with advanced retrieval strategies

---

### 📕 04 - PostgreSQL Text-to-SQL
**Level:** 🟡 Intermediate | **Duration:** ⏱️ 30-45 minutes

**Topics:**
- ☁️ Supabase cloud database integration
- 🔐 Secure connection setup with SSL
- 🔑 URL encoding for passwords with special characters
- 🔎 Schema inspection of cloud databases
- 💬 Basic querying with `NLSQLTableQueryEngine`

**💡 What you'll learn:** How to connect text-to-SQL systems to cloud PostgreSQL databases (Supabase)

**📦 Includes:**
- `create_sample_tables.py` - Generates 1,500 rows of realistic e-commerce data (500 customers, 500 products, 500 orders)
- `test_supabase_connection.py` - Verifies Supabase connection and displays database info

**📝 Note:** This notebook focuses on teaching connection patterns and basic usage. Production deployment patterns (monitoring, rate limiting, advanced error handling) are mentioned as reference material but not fully implemented.

---

## 📁 Project Structure

```
text2sql-rag/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
│
├── 01_basic_text_to_sql.ipynb        # Beginner: Basic concepts
├── 02_intermediate_text_to_sql.ipynb # Intermediate: Dynamic retrieval
├── 03_advanced_text_to_sql.ipynb     # Advanced: Workflows
├── 04_postgresql_text_to_sql.ipynb   # Intermediate: Cloud database integration
│
├── create_sample_tables.py           # Supabase data generator
└── test_supabase_connection.py       # Supabase connection tester
```

## 🎓 Learning Path

### 🎯 Recommended Order

1. **📘 Start with Notebook 01** (Basic)
   - Learn fundamental text-to-SQL concepts
   - Understand how LlamaIndex works with SQL databases
   - ⏱️ ~30-45 minutes

2. **📗 Progress to Notebook 02** (Intermediate)
   - Learn dynamic table retrieval
   - Work with multi-table databases
   - ⏱️ ~45-60 minutes

3. **📙 Advance to Notebook 03** (Advanced)
   - Master LlamaIndex Workflows
   - Implement production patterns
   - ⏱️ ~60-90 minutes

4. **📕 Connect to Cloud Databases with Notebook 04** (PostgreSQL/Supabase - Optional)
   - Learn cloud database integration
   - Understand connection security basics
   - ⏱️ ~30-45 minutes + Supabase setup

### 💡 Alternative Paths

**🚀 For experienced developers:**
- Jump to Notebook 02 or 03 directly
- Review Notebook 01 for LlamaIndex-specific concepts

**☁️ For cloud database integration:**
- Complete Notebooks 01-03 first
- Use Notebook 04 to learn PostgreSQL/Supabase connections

## ✨ Features

### 🎁 What's Included

- 📚 **4 Progressive Notebooks**: From basics to advanced workflows and cloud integration
- 🗄️ **Multiple Database Types**: SQLite, DuckDB, PostgreSQL/Supabase
- 📊 **Real-world Datasets**: E-commerce, WikiTableQuestions, employee data
- 🛡️ **Production Patterns**: Error handling examples, SQL validation patterns (Notebook 03)
- 🔍 **Vector Search**: Semantic table and row retrieval
- ☁️ **Cloud Database Integration**: PostgreSQL/Supabase connection examples
- 🎲 **Sample Data Generator**: 1,500 rows of realistic test data

### 🔒 Security Features

**✅ Implemented in notebooks:**
- 🔐 Environment variable management for API keys and credentials
- ✅ SQL query validation examples (Notebook 03)
- 🚫 Dangerous keyword blocking patterns (Notebook 03)
- 🔒 SSL connection setup for cloud databases (Notebook 04)

**📖 Discussed as best practices:**
- 👤 Read-only database roles (reference examples)
- 🔄 Connection pooling patterns
- ⏱️ Query timeout configuration

## ☁️ Supabase Setup (Notebook 04)

### ⚡ Quick Setup

1. **🌐 Create Supabase Account**
   - Go to [supabase.com](https://supabase.com)
   - Create a new project

2. **🔑 Get Connection Details**
   - Go to Project Settings → Database
   - Copy connection string details to `.env`

3. **📊 Generate Sample Data**
   ```bash
   python create_sample_tables.py
   ```

4. **✅ Verify Connection**
   ```bash
   python test_supabase_connection.py
   ```

5. **🚀 Run Notebook 04**
   - Open `04_postgresql_text_to_sql.ipynb`
   - Run cells 2, 4, 6, 11, 15, 17
   - Start querying with natural language! 🎉

### 💬 Sample Queries (Notebook 04)

```python
# Customer queries
query_engine.query("How many customers are there?")
query_engine.query("Show me customers from New York")

# Product queries
query_engine.query("What are the top 10 most expensive products?")
query_engine.query("Which products are low in stock?")

# Order queries
query_engine.query("What is the total revenue from all orders?")
query_engine.query("Which customer placed the most orders?")

# Complex queries
query_engine.query("Show me customers who ordered Electronics products")
query_engine.query("What is the most popular product category?")
```

## 🔧 Troubleshooting

### ⚠️ Common Issues

#### 📦 Import Errors
```bash
pip install --upgrade llama-index-core llama-index-llms-openai llama-index-embeddings-openai
```

#### 🔑 API Key Not Found
- ✅ Verify `.env` file is in project root
- ✅ Check `OPENAI_API_KEY` is set without quotes
- 🔄 Restart Jupyter kernel after editing `.env`
- 🧪 Test loading:
  ```python
  from dotenv import load_dotenv
  import os
  load_dotenv()
  print(os.getenv("OPENAI_API_KEY"))
  ```

#### 🗄️ Database Connection Errors
- ✅ Ensure proper file permissions for SQLite databases
- 🔍 For PostgreSQL: Verify connection string format
- 🔒 Check SSL settings for Supabase (use `sslmode=require`)
- 🧪 Test with `test_supabase_connection.py` for Notebook 04

#### ☁️ Supabase Connection Issues
- ✅ Verify all credentials in `.env` are correct
- ⚠️ Ensure no extra spaces in password
- 🟢 Check Supabase project is active
- 🔑 Use URL encoding for passwords with special characters

#### 💾 Out of Memory
- 📉 Reduce `similarity_top_k` parameter in retrievers
- 📊 Use smaller datasets for testing
- 🗑️ Close unused notebooks
- 🔄 Restart Jupyter kernel

## 💰 Cost Estimates

Using `gpt-4o-mini` (recommended):
- 📘 **Notebook 01**: ~$0.01-0.02 per run
- 📗 **Notebook 02**: ~$0.02-0.05 per run
- 📙 **Notebook 03**: ~$0.05-0.10 per run (sample data)
- 📕 **Notebook 04**: ~$0.10-0.20 per run

💵 **Total cost for completing all notebooks: ~$0.20-0.40**

## 🚀 Production Deployment Checklist

This checklist provides guidance for deploying text-to-SQL systems to production. The notebooks demonstrate some patterns (particularly in Notebook 03), but full production deployment requires additional implementation:

**⚠️ Before deploying to production:**

### 🔒 Security
- [ ] Use read-only database connections
- [ ] Implement query validation
- [ ] Set query timeouts
- [ ] Use database roles with minimal privileges
- [ ] Monitor and log all queries
- [ ] Add authentication/authorization

### ⚡ Performance
- [ ] Implement caching for common queries
- [ ] Set appropriate `similarity_top_k` values
- [ ] Use connection pooling
- [ ] Monitor query execution times
- [ ] Add rate limiting

### 🛡️ Reliability
- [ ] Implement retry logic with exponential backoff
- [ ] Add comprehensive error handling
- [ ] Set up monitoring and alerting
- [ ] Test with production data volumes
- [ ] Create backup strategies

### 💰 Cost Management
- [ ] Monitor LLM API usage
- [ ] Implement request caching
- [ ] Use appropriate model sizes
- [ ] Set usage limits and alerts

## 📚 Resources

### 📖 Official Documentation
- 🦙 [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- 🔍 [Text-to-SQL Guide](https://developers.llamaindex.ai/python/examples/index_structs/struct_indices/sqlindexdemo/)
- 🤖 [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- ☁️ [Supabase Documentation](https://supabase.com/docs)

### 📝 Tutorials
- 🗄️ [SQLAlchemy Core Tutorial](https://docs.sqlalchemy.org/en/20/core/tutorial.html)
- 🦆 [DuckDB Documentation](https://duckdb.org/docs/)
- ⚙️ [LlamaIndex Workflows](https://docs.llamaindex.ai/en/stable/understanding/workflows/)

### 👥 Community
- 💬 [LlamaIndex Discord](https://discord.gg/dGcwcsnxhU)
- 🐙 [LlamaIndex GitHub](https://github.com/run-llama/llama_index)

## ❓ FAQ

**🔄 Q: Can I use a different LLM provider?**
A: Yes! LlamaIndex supports multiple LLM providers (Anthropic Claude, Google Gemini, etc.). See [LLM integrations documentation](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/).

**💰 Q: How much does this cost to run?**
A: Using `gpt-4o-mini`, expect ~$0.20-0.40 total for all notebooks. Production costs depend on query volume.

**🗄️ Q: Can I use this with my own database?**
A: Absolutely! The patterns work with any SQLAlchemy-compatible database (MySQL, PostgreSQL, Oracle, etc.).

**🚀 Q: Is this production-ready?**
A: The notebooks teach core concepts and demonstrate key patterns (especially Notebook 03 for workflows and error handling). For full production deployment, you'll need to implement additional features from the deployment checklist, such as comprehensive monitoring, rate limiting, and advanced error handling. Notebook 04 shows cloud database connection patterns but focuses on teaching rather than production implementation.

**📊 Q: How do I handle large databases with hundreds of tables?**
A: Use dynamic table retrieval (Notebooks 02 & 03), implement database views for complex schemas, and adjust `similarity_top_k` for retrieval.

**🌐 Q: Can I deploy this as a web API?**
A: Yes! Use FastAPI or Flask to wrap the query engine. Notebooks 03 and 04 demonstrate workflow patterns and database connections that can serve as foundations for a web API.

## 🤝 Contributing

Contributions are welcome! If you find issues or have suggestions:

1. ✅ Check existing issues first
2. 📝 Provide detailed descriptions
3. 🐛 Include error messages and environment details
4. 🧪 Test changes with all notebooks

## 📄 License

MIT License - feel free to use this project for learning and commercial purposes!

## 🙏 Acknowledgments

- 🦙 Built with [LlamaIndex](https://github.com/run-llama/llama_index)
- 🤖 Powered by OpenAI GPT models
- ☁️ Cloud database integration with [Supabase](https://supabase.com)
- 💡 Inspired by [LlamaIndex documentation examples](https://developers.llamaindex.ai/)

---

<div align="center">

## 🎉 Happy Learning! 🚀

**If you find these notebooks helpful, please consider ⭐ starring the repository and sharing with others!**

Made with ❤️ by the community

</div>
