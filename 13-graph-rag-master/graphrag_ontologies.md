# GraphRAG Ontologies & Knowledge Graph Construction
### Two Use Cases: Financial Documents · Social Media

> **Reference Architecture:** Each use case is broken into three layers —
> **(1) Document Ontology** → **(2) Attribute Schemas** → **(3) Knowledge Graph Build Pipeline**

---

---

# 📊 USE CASE 1 — Financial Documents

> **Domain:** SEC filings, earnings reports, analyst notes, press releases, regulatory disclosures

---

## 1.1 — Document Ontology (Financial)

Defines the **node types** and **edge types** that govern what can exist in the graph.
The LLM can only extract what is defined here — nothing more.

```mermaid
graph TD
    DOC["📄 Document\n(10-K / 10-Q / Press Release\nEarnings Call / Analyst Report)"]
    CHUNK["📋 Chunk\n(Section / Paragraph)"]
    COMPANY["🏢 Company\n(Issuer / Competitor / Subsidiary)"]
    PERSON["👤 Executive\n(CEO / CFO / Board Member)"]
    METRIC["📈 FinancialMetric\n(Revenue / EPS / EBITDA / Ratio)"]
    EVENT["⚡ CorporateEvent\n(Merger / Acquisition / Earnings / IPO)"]
    RISK["⚠️ RiskFactor\n(Regulatory / Market / Operational)"]
    REGULATION["⚖️ Regulation\n(SEC Rule / GAAP Standard / Law)"]

    DOC -->|"PART_OF"| CHUNK
    CHUNK -->|"NEXT"| CHUNK
    DOC -->|"REFERENCES"| DOC
    DOC -->|"FILED_BY"| COMPANY
    DOC -->|"MENTIONS"| PERSON
    DOC -->|"DISCLOSES"| METRIC
    DOC -->|"DESCRIBES"| EVENT
    DOC -->|"CONTAINS"| RISK
    RISK -->|"GOVERNED_BY"| REGULATION
    PERSON -->|"LEADS"| COMPANY
    COMPANY -->|"SUBSIDIARY_OF"| COMPANY
    COMPANY -->|"COMPETITOR_OF"| COMPANY
    EVENT -->|"INVOLVES"| COMPANY
    METRIC -->|"REPORTED_BY"| COMPANY
    METRIC -->|"COMPARED_TO"| METRIC

    style DOC fill:#1a3a5c,color:#fff,stroke:#4a90d9
    style CHUNK fill:#1a3a5c,color:#fff,stroke:#4a90d9
    style COMPANY fill:#2d6a4f,color:#fff,stroke:#52b788
    style PERSON fill:#2d6a4f,color:#fff,stroke:#52b788
    style METRIC fill:#7b2d00,color:#fff,stroke:#e76f51
    style EVENT fill:#7b2d00,color:#fff,stroke:#e76f51
    style RISK fill:#4a1942,color:#fff,stroke:#c77dff
    style REGULATION fill:#4a1942,color:#fff,stroke:#c77dff
```

### Node & Edge Type Registry

```mermaid
graph LR
    subgraph NODE_TYPES["✅ Allowed Node Types (6)"]
        N1["Document"]
        N2["Chunk"]
        N3["Company"]
        N4["Executive"]
        N5["FinancialMetric"]
        N6["CorporateEvent"]
        N7["RiskFactor"]
        N8["Regulation"]
    end

    subgraph EDGE_TYPES["✅ Allowed Edge Types (10)"]
        E1["PART_OF"]
        E2["NEXT"]
        E3["REFERENCES"]
        E4["FILED_BY"]
        E5["MENTIONS / LEADS"]
        E6["DISCLOSES / REPORTED_BY"]
        E7["DESCRIBES / INVOLVES"]
        E8["CONTAINS"]
        E9["GOVERNED_BY"]
        E10["SUBSIDIARY_OF / COMPETITOR_OF"]
    end

    style NODE_TYPES fill:#1a3a5c,color:#fff
    style EDGE_TYPES fill:#2d6a4f,color:#fff
```

---

## 1.2 — Attribute Schemas (Financial)

Defines the **properties** attached to each node type.
Metadata nodes (Document, Chunk) are populated structurally — **no LLM needed**.
Semantic nodes (Company, Executive, Metric, Event, Risk) use **LLM extraction**.

```mermaid
graph TD
    subgraph METADATA_NODES["🔵 Metadata-Driven (No LLM Cost)"]
        DOC_SCHEMA["📄 Document\n─────────────────\n• doc_id: string\n• source_uri: string\n• source_type: enum[10-K, 10-Q, 8-K,\n  EarningsCall, AnalystReport]\n• filing_date: date\n• fiscal_period: string\n• cik_number: string\n• exchange: enum[NYSE, NASDAQ, LSE]\n• language: string\n• page_count: int"]

        CHUNK_SCHEMA["📋 Chunk\n─────────────────\n• chunk_id: string\n• source_uri: string\n• content: text\n• section_title: string\n• page_number: int\n• sequence_order: int\n• date: date\n• embedding_vector: float[]"]
    end

    subgraph LLM_NODES["🟠 LLM-Extracted (Semantic)"]
        COMPANY_SCHEMA["🏢 Company\n─────────────────\n• canonical_name: string\n• ticker: string\n• aliases: string[]\n• sector: string\n• industry: string\n• headquarters: string\n• cik: string\n• founded_year: int"]

        PERSON_SCHEMA["👤 Executive\n─────────────────\n• canonical_name: string\n• aliases: string[]\n• title: string\n• email: string\n• tenure_start: date\n• tenure_end: date\n• confidence: float"]

        METRIC_SCHEMA["📈 FinancialMetric\n─────────────────\n• metric_type: enum[Revenue, EPS,\n  EBITDA, NetIncome, DebtRatio]\n• value: float\n• unit: string\n• period: string\n• yoy_change: float\n• source_chunk_id: string\n• confidence: float"]

        EVENT_SCHEMA["⚡ CorporateEvent\n─────────────────\n• event_type: enum[Merger, Acquisition,\n  Earnings, IPO, Layoff, Spin-off]\n• event_date: date\n• description: text\n• deal_value: float\n• outcome: string\n• confidence: float"]

        RISK_SCHEMA["⚠️ RiskFactor\n─────────────────\n• risk_category: enum[Market, Regulatory,\n  Operational, Liquidity, Cyber]\n• description: text\n• severity: enum[Low, Medium, High]\n• mitigation: text\n• confidence: float"]
    end

    style METADATA_NODES fill:#1a3a5c,color:#fff
    style LLM_NODES fill:#7b2d00,color:#fff
```

---

## 1.3 — Knowledge Graph Build Pipeline (Financial)

End-to-end flow from raw documents to a queryable graph.

```mermaid
flowchart TD
    A["📥 Raw Financial Documents\n10-K · 10-Q · 8-K · Earnings Calls\nAnalyst Reports · Press Releases"]

    A --> B["🔍 Document Parser\nPDF/HTML extraction\nLayout-aware chunking\n(Docling / MinerU)"]

    B --> C["📋 Metadata Extraction\nNo LLM — structural only\nPopulate: doc_id, source_uri,\nfiling_date, cik, exchange"]

    B --> D["✂️ Semantic Chunking\nSection-aware splits\nOverlapping windows\nEmbed each chunk"]

    C --> E["🗄️ Document Collection\nMongoDB / raw store\nImmutable append-only"]
    D --> E

    D --> F["🤖 Constrained LLM Extraction\nOntology injected into prompt\nExtract: Company, Executive,\nMetric, Event, RiskFactor\nReturn surface_form + canonical_name\n+ type + confidence"]

    F --> G["🔎 Entity Resolution Pipeline\nStage 1: Registry lookup\nStage 2: Fuzzy match RapidFuzz\nStage 3: Embedding cosine sim\nStage 4: LLM arbitration\nfor ambiguous cases only"]

    G --> H["📝 Observation Log\nImmutable extraction store\nLinks surface_form → canonical\nwith doc_id + chunk_id + confidence"]

    H --> I{"Duplicate\nDetected?"}

    I -->|"Yes — merge"| J["🔗 Merge to Existing Node\nUpdate aliases list\nIncrement mention_count\nAdd provenance link"]

    I -->|"No — new entity"| K["✨ Create New Node\nAssign canonical_name\nSet attributes from schema\nGenerate embedding"]

    J --> L["🏗️ Materialize Graph\nMongoDB $aggregation\nBuild edge collection\nfrom observation logs"]
    K --> L

    L --> M["📊 Graph Quality Checks\nSchema violation rate\nOrphan node ratio\nDuplicate node ratio\nEdge type distribution\nSchema drift score"]

    M --> N{"Quality\nThreshold\nMet?"}

    N -->|"No"| O["🔧 Flag for Review\nSurface violations\nTighten ontology\nRetrain extraction prompt"]
    O --> F

    N -->|"Yes"| P["✅ Queryable Knowledge Graph\nText Search → $vectorSearch\n→ RRF → $graphLookup"]

    P --> Q["🔍 Query Patterns\nMulti-hop: Company → Metric → Event\nRisk chain: RiskFactor → Regulation\nExecutive graph: Person → Company → Event\nPeer comparison: Company → COMPETITOR_OF → Company"]

    style A fill:#1a3a5c,color:#fff
    style F fill:#7b2d00,color:#fff
    style G fill:#2d6a4f,color:#fff
    style L fill:#4a1942,color:#fff
    style P fill:#1a3a5c,color:#fff
    style Q fill:#2d6a4f,color:#fff
    style O fill:#7b2d00,color:#fff
```

---

---

# 📱 USE CASE 2 — Social Media

> **Domain:** Tweets/X posts, LinkedIn posts, Reddit threads, Instagram captions, YouTube comments, blog posts

---

## 2.1 — Document Ontology (Social Media)

```mermaid
graph TD
    POST["📱 Post\n(Tweet / LinkedIn / Reddit\nInstagram / YouTube Comment)"]
    THREAD["🧵 Thread\n(Conversation chain\n/ Comment tree)"]
    PROFILE["👤 Profile\n(User / Creator / Brand)"]
    TOPIC["🏷️ Topic\n(Hashtag / Theme / Trend)"]
    ENTITY["🔍 NamedEntity\n(Person / Brand / Product\nEvent / Location)"]
    SENTIMENT["💬 Sentiment\n(Opinion / Reaction\n/ Stance)"]
    CAMPAIGN["📣 Campaign\n(Viral thread / Ad campaign\n/ Coordinated push)"]
    COMMUNITY["🌐 Community\n(Subreddit / Group\n/ Niche audience)"]

    POST -->|"PART_OF"| THREAD
    POST -->|"NEXT"| POST
    POST -->|"REPLY_TO"| POST
    POST -->|"AUTHORED_BY"| PROFILE
    POST -->|"TAGGED_WITH"| TOPIC
    POST -->|"MENTIONS"| ENTITY
    POST -->|"EXPRESSES"| SENTIMENT
    POST -->|"BELONGS_TO"| CAMPAIGN
    PROFILE -->|"FOLLOWS"| PROFILE
    PROFILE -->|"MEMBER_OF"| COMMUNITY
    PROFILE -->|"CREATES"| POST
    ENTITY -->|"ASSOCIATED_WITH"| TOPIC
    SENTIMENT -->|"TARGETS"| ENTITY
    TOPIC -->|"RELATED_TO"| TOPIC
    CAMPAIGN -->|"TARGETS"| COMMUNITY

    style POST fill:#0d3b66,color:#fff,stroke:#4cc9f0
    style THREAD fill:#0d3b66,color:#fff,stroke:#4cc9f0
    style PROFILE fill:#1b4332,color:#fff,stroke:#52b788
    style TOPIC fill:#1b4332,color:#fff,stroke:#52b788
    style ENTITY fill:#6d2b3d,color:#fff,stroke:#f72585
    style SENTIMENT fill:#6d2b3d,color:#fff,stroke:#f72585
    style CAMPAIGN fill:#3d1a78,color:#fff,stroke:#7b2fff
    style COMMUNITY fill:#3d1a78,color:#fff,stroke:#7b2fff
```

### Node & Edge Type Registry

```mermaid
graph LR
    subgraph NODE_TYPES["✅ Allowed Node Types (8)"]
        N1["Post"]
        N2["Thread"]
        N3["Profile"]
        N4["Topic"]
        N5["NamedEntity"]
        N6["Sentiment"]
        N7["Campaign"]
        N8["Community"]
    end

    subgraph EDGE_TYPES["✅ Allowed Edge Types (12)"]
        E1["PART_OF / NEXT"]
        E2["REPLY_TO"]
        E3["AUTHORED_BY / CREATES"]
        E4["TAGGED_WITH"]
        E5["MENTIONS"]
        E6["EXPRESSES / TARGETS"]
        E7["BELONGS_TO"]
        E8["FOLLOWS"]
        E9["MEMBER_OF"]
        E10["ASSOCIATED_WITH"]
        E11["RELATED_TO"]
        E12["TARGETS (Campaign→Community)"]
    end

    style NODE_TYPES fill:#0d3b66,color:#fff
    style EDGE_TYPES fill:#1b4332,color:#fff
```

---

## 2.2 — Attribute Schemas (Social Media)

```mermaid
graph TD
    subgraph METADATA_NODES["🔵 Metadata-Driven (No LLM Cost)"]
        POST_SCHEMA["📱 Post\n─────────────────\n• post_id: string\n• source_uri: string\n• source_type: enum[Tweet, LinkedIn,\n  Reddit, Instagram, YouTube]\n• content: text\n• created_at: datetime\n• language: string\n• like_count: int\n• share_count: int\n• reply_count: int\n• is_verified_author: bool\n• embedding_vector: float[]"]

        THREAD_SCHEMA["🧵 Thread\n─────────────────\n• thread_id: string\n• platform: string\n• root_post_id: string\n• depth: int\n• participant_count: int\n• created_at: datetime\n• total_engagement: int"]
    end

    subgraph LLM_NODES["🟠 LLM-Extracted (Semantic)"]
        PROFILE_SCHEMA["👤 Profile\n─────────────────\n• canonical_handle: string\n• display_name: string\n• aliases: string[]\n• platform: string\n• account_type: enum[Person,\n  Brand, Bot, Media]\n• follower_tier: enum[Nano,\n  Micro, Macro, Mega]\n• verified: bool\n• bio_summary: text"]

        TOPIC_SCHEMA["🏷️ Topic\n─────────────────\n• canonical_topic: string\n• hashtags: string[]\n• aliases: string[]\n• topic_category: enum[Tech, Finance,\n  Politics, Entertainment, Sports]\n• trend_score: float\n• first_seen: date\n• peak_volume_date: date"]

        ENTITY_SCHEMA["🔍 NamedEntity\n─────────────────\n• canonical_name: string\n• entity_type: enum[Person, Brand,\n  Product, Event, Location]\n• aliases: string[]\n• description: text\n• confidence: float\n• wiki_url: string"]

        SENTIMENT_SCHEMA["💬 Sentiment\n─────────────────\n• sentiment_label: enum[Positive,\n  Negative, Neutral, Mixed]\n• sentiment_score: float\n• emotion: enum[Joy, Anger,\n  Fear, Surprise, Disgust]\n• stance: enum[Support, Oppose,\n  Neutral, Sarcastic]\n• confidence: float"]

        CAMPAIGN_SCHEMA["📣 Campaign\n─────────────────\n• campaign_id: string\n• campaign_name: string\n• campaign_type: enum[Organic,\n  Paid, Coordinated, Viral]\n• start_date: date\n• end_date: date\n• reach_estimate: int\n• primary_topic: string"]
    end

    style METADATA_NODES fill:#0d3b66,color:#fff
    style LLM_NODES fill:#6d2b3d,color:#fff
```

---

## 2.3 — Knowledge Graph Build Pipeline (Social Media)

```mermaid
flowchart TD
    A["📥 Raw Social Media Data\nTweets · LinkedIn Posts · Reddit Threads\nInstagram Captions · YouTube Comments"]

    A --> B["🔌 Platform Ingestion Layer\nAPI connectors per platform\nRate-limit aware polling\nWebhook / streaming ingestion"]

    B --> C["🧹 Pre-processing\nDe-duplication by post_id\nURL normalization\nEmoji → text normalization\nLanguage detection"]

    C --> D["📋 Metadata Population\nNo LLM — structural only\npost_id, source_uri, platform\ncreated_at, engagement counts\nthread reconstruction"]

    C --> E["✂️ Content Chunking\nPosts are atomic units\nThreads chunked by reply depth\nEmbed each post independently"]

    D --> F["🗄️ Raw Post Collection\nAppend-only store\nPreserve original content\n+ all metadata"]
    E --> F

    E --> G["🤖 Constrained LLM Extraction\nOntology injected into prompt\nExtract: Profile, Topic, NamedEntity\nSentiment, Campaign signals\nConstrain to defined enum values only"]

    G --> H["😄 Sentiment Analysis Layer\nDedicated sentiment model\n(cheaper than frontier LLM)\nLabel: sentiment + emotion + stance\nHandle sarcasm detection separately"]

    H --> I["🔎 Entity Resolution\nHandle: @mentions → Profile\n#hashtags → Topic canonicalization\nBrand name variants → NamedEntity\nFuzzy + embedding resolution"]

    I --> J["📝 Observation Log\nImmutable extraction store\npost_id + entity + type\n+ confidence + timestamp"]

    J --> K{"Duplicate\nProfile or\nEntity?"}

    K -->|"Yes — merge"| L["🔗 Merge Node\nConsolidate aliases\nAggregate sentiment history\nUpdate mention_count"]

    K -->|"No — new"| M["✨ Create New Node\nSet canonical attributes\nGenerate embedding\nAssign to community if known"]

    L --> N["🏗️ Materialize Graph\nBuild edges from observations\nPOST→AUTHORED_BY→PROFILE\nPOST→TAGGED_WITH→TOPIC\nPOST→EXPRESSES→SENTIMENT\nPOST→MENTIONS→ENTITY"]
    M --> N

    N --> O["📊 Graph Quality Checks\nBot account detection\nTopic drift monitoring\nSentiment distribution audit\nCross-platform entity alignment\nSchema violation rate"]

    O --> P{"Quality\nThreshold\nMet?"}

    P -->|"No"| Q["🔧 Flag & Remediate\nReview bot-like profiles\nMerge fragmented topics\nRetighten ontology constraints"]
    Q --> G

    P -->|"Yes"| R["✅ Queryable Social Graph"]

    R --> S["🔍 Query Patterns\nInfluencer mapping: Profile → FOLLOWS → Profile\nNarrative tracking: Topic → RELATED_TO → Topic\nSentiment chain: Entity → TARGETS → Sentiment over time\nCampaign attribution: Campaign → BELONGS_TO → Post → AUTHORED_BY → Profile\nCommunity detection: Profile → MEMBER_OF → Community → TARGETS"]

    style A fill:#0d3b66,color:#fff
    style G fill:#6d2b3d,color:#fff
    style H fill:#6d2b3d,color:#fff
    style I fill:#1b4332,color:#fff
    style N fill:#3d1a78,color:#fff
    style R fill:#0d3b66,color:#fff
    style S fill:#1b4332,color:#fff
    style Q fill:#6d2b3d,color:#fff
```

---

---

# 🔁 Cross-Use-Case Comparison

```mermaid
graph TD
    subgraph FINANCIAL["📊 Financial Documents"]
        F_ANCHOR["Core Anchor Node: Company"]
        F_SIGNAL["Primary Signal: FinancialMetric"]
        F_RISK["Risk Layer: RiskFactor → Regulation"]
        F_QUERY["Key Query: Multi-hop financial dependency chains"]
        F_CHEAP["Cheap Model Safe For: Metric extraction\n(structured, low ambiguity)"]
    end

    subgraph SOCIAL["📱 Social Media"]
        S_ANCHOR["Core Anchor Node: Profile"]
        S_SIGNAL["Primary Signal: Sentiment"]
        S_RISK["Risk Layer: Campaign → Community manipulation"]
        S_QUERY["Key Query: Narrative spread + influence mapping"]
        S_CHEAP["Cheap Model Safe For: Topic tagging\n(hashtag-driven, low ambiguity)"]
    end

    subgraph SHARED["🔗 Shared Principles"]
        SP1["Ontology-first — define before extracting"]
        SP2["Metadata nodes never go through LLM"]
        SP3["Immutable observation log always present"]
        SP4["Entity resolution is a pipeline not a step"]
        SP5["Schema drift score monitored weekly"]
        SP6["Multi-hop queries validate graph value"]
    end

    FINANCIAL --- SHARED
    SOCIAL --- SHARED

    style FINANCIAL fill:#1a3a5c,color:#fff
    style SOCIAL fill:#0d3b66,color:#fff
    style SHARED fill:#2d6a4f,color:#fff
```

---