# Neo4j & Cypher: Step-by-Step Browser Guide

> **How to use this guide:**
> 1. Open **Neo4j Browser** (see Section 2 for setup options)
> 2. Paste each `cypher` code block into the editor at the top
> 3. Press **Ctrl+Enter** (Windows/Linux) or **Cmd+Enter** (Mac) — or click the ▶ Run button
> 4. Follow along section by section — every command builds on the last
> 5. Presenter notes are marked with 🎤 — these are talking points, not commands

---

## Table of Contents

1. [What is Neo4j?](#1-what-is-neo4j)
2. [Opening Neo4j Browser](#2-opening-neo4j-browser)
3. [Step 1 — Verify Connection](#3-step-1--verify-connection)
4. [Step 2 — CREATE Nodes](#4-step-2--create-nodes)
5. [Step 3 — CREATE Relationships](#5-step-3--create-relationships)
6. [Step 4 — MATCH & Query](#6-step-4--match--query)
7. [Step 5 — MERGE (Upsert)](#7-step-5--merge-upsert)
8. [Step 6 — UPDATE Properties](#8-step-6--update-properties-set)
9. [Step 7 — DELETE Nodes & Relationships](#9-step-7--delete-nodes-and-relationships)

---

## 1. What is Neo4j?

Neo4j is a **native graph database** that stores data as nodes, relationships, and properties — instead of tables and rows. It uses the **labeled property graph** (LPG) model, where every entity is a node with a label, every connection is a typed, directed relationship, and both can carry arbitrary key-value properties. You query it using **Cypher**, a declarative, pattern-matching language designed to feel like drawing the graph you want to find.

### The Labeled Property Graph Model

```
     ┌─────────────────────────────────┐
     │           NODE                  │
     │  Label:  :Person                │
     │  Props:  name = "Alice"         │
     │          age  = 30              │
     │          role = "Data Scientist"│
     └──────────────┬──────────────────┘
                    │
          RELATIONSHIP (directed)
          Type:   :WATCHED
          Props:  rating     = 5
                  watched_on = "2024-01-15"
                    │
     ┌──────────────▼──────────────────┐
     │           NODE                  │
     │  Label:  :Movie                 │
     │  Props:  title  = "The Matrix"  │
     │          year   = 1999          │
     │          rating = 8.7           │
     └─────────────────────────────────┘
```

**Four building blocks:**

| Concept | Symbol | Description |
|---------|--------|-------------|
| **Node** | `( )` | An entity — Person, Movie, Product |
| **Label** | `:Person` | A type tag on a node (can have multiple) |
| **Relationship** | `-[:TYPE]->` | A directed, typed connection between nodes |
| **Property** | `{key: value}` | Attribute on a node or relationship |

### SQL vs Cypher — A Quick Contrast

**SQL** (relational — finding friends of friends requires JOINs):
```sql
SELECT u2.name
FROM users u1
JOIN friendships f ON u1.id = f.user_id
JOIN users u2 ON f.friend_id = u2.id
WHERE u1.name = 'Alice';
```

**Cypher** (graph — traverse the pattern naturally):
```cypher
MATCH (alice:Person {name: "Alice"})-[:KNOWS]->(friend:Person)
RETURN friend.name
```

### When to Use Graph vs Relational

| Use Case | Graph DB ✅ | Relational DB ✅ |
|----------|------------|-----------------|
| Social networks / recommendations | ✅ Natural traversal | ❌ Deep JOINs get slow |
| Fraud detection (connected rings) | ✅ Pattern matching | ❌ Hard to express |
| Knowledge graphs / RAG | ✅ Entity relationships | ❌ Rigid schema |
| Simple tabular reporting | ❌ Overhead | ✅ Fast aggregations |
| Transactional CRUD (e-commerce) | ❌ Overhead | ✅ ACID optimized |
| Path finding / routing | ✅ Built-in algorithms | ❌ Recursive CTEs |

---

## 2. Opening Neo4j Browser

Neo4j Browser is the web-based GUI where you type and run Cypher queries. Choose the option that matches your setup:

### Option A — AuraDB (Free Cloud, Recommended for Beginners)

1. Go to [console.neo4j.io](https://console.neo4j.io)
2. Sign in (free account) → click **Open** on your instance
3. Click the **Query** tab (or "Browser" tab depending on version)
4. You'll be prompted for your password — use the one from your AuraDB credentials file

### Option B — Local Docker

Run this single command in your terminal:

```bash
docker run \
  --name neo4j-demo \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:latest
```

Then open **http://localhost:7474** in your browser and log in with `neo4j` / `password123`.

### Option C — Neo4j Desktop

1. Open Neo4j Desktop app
2. Click **Start** on your database
3. Click **Open** → **Neo4j Browser**

### What the Browser Looks Like

```
┌─────────────────────────────────────────────────────────────────┐
│  Neo4j Browser                                          [★] [?] │
├─────────────────────────────────────────────────────────────────┤
│  neo4j$  ┌──────────────────────────────────────────────────┐   │
│          │  MATCH (n) RETURN n                           ▶  │   │
│          └──────────────────────────────────────────────────┘   │
│                    ↑ Type Cypher here. Press ▶ or Ctrl+Enter    │
├─────────────────────────────────────────────────────────────────┤
│  Results appear here                                            │
│  [Graph] [Table] [Text] [Code]  ← toggle result views          │
│                                                                 │
│     (Alice) ──[:WATCHED]──► (The Matrix)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

> 🎤 **Presenter note:** The **Graph** tab shows a visual node-link diagram. The **Table** tab shows results as rows. Switch between them to show how the same data looks different ways.

---

## 3. Step 1 — Verify Connection

Before anything else, confirm the database is alive and Cypher works.

```cypher
RETURN "Hello, Neo4j!" AS message, 1+1 AS math
```

> **Expected output (Table view):**
> | message | math |
> |---------|------|
> | Hello, Neo4j! | 2 |

> 🎤 **Presenter note:** This query has no MATCH — it just evaluates expressions. If you see this table, your connection is working and you're ready to go.

---

## 4. Step 2 — CREATE Nodes

Nodes are the entities in your graph. Let's build up a small social + movie graph step by step.

### 4a. Create a Single Person Node

```cypher
CREATE (p:Person {name: "Alice", age: 30, role: "Data Scientist"})
RETURN p
```

> **Expected output (Table view):**
> ```
> p
> ───────────────────────────────────────────────
> (:Person {name: "Alice", age: 30, role: "Data Scientist"})
> ```
> In **Graph view**, you'll see a single circle labelled "Alice".

> 🎤 `CREATE` writes to the database. `(p:Person {...})` declares a node pattern: `p` is a variable, `:Person` is the label, `{...}` are the properties. `RETURN p` sends the node back so we can see it.

### 4b. Create Multiple Nodes in One Query

```cypher
CREATE
  (b:Person {name: "Bob",     age: 28, role: "Engineer"}),
  (c:Person {name: "Charlie", age: 35, role: "Designer"})
RETURN b, c
```

> **Expected output:** Two nodes returned — Bob and Charlie. Graph view shows two circles.

> 🎤 You can comma-separate multiple patterns inside a single `CREATE`. All are written in one atomic operation.

### 4c. Create a Movie Node

```cypher
CREATE (m:Movie {title: "The Matrix", year: 1999, genre: "Sci-Fi", rating: 8.7})
RETURN m
```

> **Expected output:** A single Movie node with all four properties shown.

> 🎤 Notice `:Movie` is a different label from `:Person`. Labels let us distinguish entity types and filter queries efficiently — they work like type tags.

### 4d. See the Full Graph So Far

```cypher
MATCH (n) RETURN n
```

> **Expected output (Graph view):** Four circles — Alice, Bob, Charlie, and The Matrix — but no lines connecting them yet. That's what Step 3 is for.

> 🎤 `MATCH (n)` with no filter matches every node in the database. Switch to **Graph** tab — you'll see all four nodes floating separately. Properties appear when you click a node.

---

## 5. Step 3 — CREATE Relationships

Relationships are the edges — the *connections* that make a graph meaningful.

### 5a. Relationship Syntax

```
(a:Label) -[:RELATIONSHIP_TYPE {prop: value}]-> (b:Label)
  ↑ start node     ↑ type + direction + props         ↑ end node
```

- Relationships are **directed** (arrow shows direction)
- The type is always **UPPERCASE_WITH_UNDERSCORES** by convention
- Both nodes and relationships can carry properties

### 5b. Link Alice to The Matrix

```cypher
MATCH (p:Person {name: "Alice"}), (m:Movie {title: "The Matrix"})
CREATE (p)-[:WATCHED {rating: 5, watched_on: "2024-01-15"}]->(m)
RETURN p, m
```

> **Expected output (Graph view):** Alice and The Matrix circles connected by an arrow labelled `:WATCHED`.

> 🎤 We `MATCH` the existing nodes first (find what's already there), then `CREATE` the relationship between them. The relationship carries its own properties: `rating` and `watched_on`.

### 5c. Create a KNOWS Relationship Between People

```cypher
MATCH (a:Person {name: "Alice"}), (b:Person {name: "Bob"})
CREATE (a)-[:KNOWS {since: 2020}]->(b)
RETURN a, b
```

> **Expected output (Graph view):** Alice → Bob arrow labelled `:KNOWS {since: 2020}`.

### 5d. Visualize the Full Graph

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m
```

> **Expected output (Graph view):** All nodes and relationships visible — Alice connected to The Matrix via `:WATCHED`, and Alice connected to Bob via `:KNOWS`. Charlie is still isolated (no relationships yet).

> 🎤 Click **Graph** tab. You can drag nodes around, click them to see properties, and double-click to expand neighbors. This visual is what makes Neo4j compelling for demos.

---

## 6. Step 4 — MATCH & Query

`MATCH` is how you read from the graph. Think of it as `SELECT` + pattern matching in one.

### 6a. Return All Nodes

```cypher
MATCH (n) RETURN n
```

> **What it does:** Returns every node in the database — all labels, all properties.

### 6b. Filter by Label and Property

```cypher
MATCH (p:Person) WHERE p.age > 28 RETURN p.name, p.age, p.role
```

> **Expected output (Table view):**
> | p.name | p.age | p.role |
> |--------|-------|--------|
> | Alice | 30 | Data Scientist |
> | Charlie | 35 | Designer |

> **What it does:** Finds only `:Person` nodes, filters to those older than 28, and returns specific properties as columns.

### 6c. Traverse a Relationship

```cypher
MATCH (p:Person)-[r:WATCHED]->(m:Movie)
RETURN p.name AS person, m.title AS movie, r.rating AS rating
```

> **Expected output (Table view):**
> | person | movie | rating |
> |--------|-------|--------|
> | Alice | The Matrix | 5 |

> **What it does:** Follows the `:WATCHED` relationship from Person to Movie and returns data from all three — the person, the movie, and the relationship property `rating`. The `AS` keyword renames columns.

### 6d. ORDER BY and LIMIT

```cypher
MATCH (p:Person)-[r:WATCHED]->(m:Movie)
RETURN p.name, m.title, r.rating
ORDER BY r.rating DESC
LIMIT 5
```

> **What it does:** Same traversal, but sorted by rating (highest first) and capped at 5 results. Useful when the graph gets large.

### 6e. Multi-Hop Traversal (Friends-of-Friends Pattern)

```cypher
MATCH (p:Person)-[:KNOWS]->(friend:Person)-[:WATCHED]->(m:Movie)
WHERE p.name = "Alice"
RETURN DISTINCT friend.name AS friend, m.title AS movie
```

> **What it does:** Starting from Alice, follows `:KNOWS` to her friends, then follows `:WATCHED` to movies *they* watched — discovering movies Alice hasn't seen herself but her network has. `DISTINCT` removes duplicates.

> 🎤 This two-hop traversal is the core of **collaborative filtering** (recommendation engines). In SQL this would be two JOINs with a WHERE; in Cypher it reads like the English sentence.

### 6f. Find Relationship Types Dynamically

```cypher
MATCH (a)-[r]->(b)
RETURN a.name AS from, type(r) AS relationship, b.name AS to, b.title AS title
```

> **What it does:** Returns all relationships in the graph without filtering by type. `type(r)` returns the string name of the relationship type (e.g., `"WATCHED"`, `"KNOWS"`). Useful for exploring an unfamiliar graph schema.

---

## 7. Step 5 — MERGE (Upsert)

`MERGE` = **create if not exists, match if it does**. It's the safe production alternative to `CREATE` — prevents duplicate data.

> 🎤 In production you almost always use `MERGE` instead of `CREATE`. `CREATE` will happily make a second Alice node if you run it twice. `MERGE` won't.

### 7a. MERGE with ON CREATE / ON MATCH

```cypher
MERGE (p:Person {name: "Diana"})
ON CREATE SET p.role = "Manager",   p.created_at = timestamp()
ON MATCH  SET p.last_seen = timestamp()
RETURN p
```

> **First run — Expected output:** A new Person node for Diana is created with `role` and `created_at` set.
>
> **Second run — Expected output:** The existing Diana node is found; `last_seen` is updated. No duplicate created.

> **What it does:**
> - `ON CREATE SET` — runs only when the node is **newly created**
> - `ON MATCH SET` — runs only when the node **already existed**

### 7b. MERGE a Relationship Safely

```cypher
MATCH (a:Person {name: "Alice"}), (b:Person {name: "Charlie"})
MERGE (a)-[:KNOWS]->(b)
RETURN a, b
```

> **What it does:** Creates the `:KNOWS` relationship from Alice to Charlie only if it doesn't already exist. Run it twice — only one relationship is ever created.

---

## 8. Step 6 — UPDATE Properties (SET)

Use `SET` to add or update properties on existing nodes and relationships.

### Update Node Properties

```cypher
MATCH (p:Person {name: "Alice"})
SET p.age = 31, p.role = "Senior Data Scientist"
RETURN p
```

> **Expected output:** Alice's node now shows `age: 31` and `role: "Senior Data Scientist"`. Properties are updated in-place — no duplicate node is created.

> 🎤 You can `SET` multiple properties in one statement with a comma. Existing properties are overwritten; new ones are added.

### Add a Property to a Relationship

```cypher
MATCH (:Person {name: "Alice"})-[r:WATCHED]->(:Movie {title: "The Matrix"})
SET r.rewatched = true
RETURN r
```

> **Expected output:** The `:WATCHED` relationship now has three properties: `rating`, `watched_on`, and `rewatched: true`.

> 🎤 Relationships are first-class citizens in Neo4j — they hold properties just like nodes do.

---

## 9. Step 7 — DELETE Nodes and Relationships

### 9a. Delete a Single Relationship

```cypher
MATCH (a:Person {name: "Alice"})-[r:KNOWS]->(b:Person {name: "Bob"})
DELETE r
```

> **Expected output:** `Deleted 1 relationship`. The Alice and Bob nodes remain; only the `:KNOWS` edge between them is removed.

### 9b. Delete a Node (DETACH DELETE)

```cypher
MATCH (p:Person {name: "Charlie"})
DETACH DELETE p
```

> **Expected output:** `Deleted 1 node, deleted N relationships`. Charlie is gone along with all relationships attached to him.

> 🎤 Plain `DELETE` on a node with relationships will throw an error — Neo4j protects you from leaving dangling edges. `DETACH DELETE` removes the node **and** all its relationships atomically. In production, prefer this.

### 9c. Clear the Entire Database (Demo Reset)

> ⚠️ **WARNING: This deletes all data permanently. Only use to reset your demo database.**

```cypher
MATCH (n) DETACH DELETE n
```

> **Expected output:** `Deleted N nodes, deleted M relationships`. The database is now empty.

> 🎤 You'll use this at the start of the full Movie Graph demo in Section 12 to start fresh.

---

