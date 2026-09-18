# AIMLite: Licensing Strategy & Commercialization Model

This document outlines the commercial strategy, dual-licensing structure, and sustainable monetization pathways for the AIMLite framework. It defines the boundary between the **forever-free Community Edition** and the **commercial enterprise tier**, balancing open-source viral adoption with high-margin enterprise revenue.

---

## 1. Core Philosophy: The Apache 2.0 Open-Source Guarantee

To maximize grassroots adoption across developers, students, researchers, and startups, AIMLite is licensed under the permissive **Apache License 2.0**:

### The Apache 2.0 Guarantee:
- **100% Free Forever**: Available directly via `pip install aimlite` or `uv add aimlite`.
- **Zero Legal Anxiety**: No copyleft restrictions, no AGPL viral clauses. Developers and companies can use, modify, and embed AIMLite in commercial or non-commercial software freely.
- **Fully Featured**: Complete access to all 3 core pillars (Data, Model, Lifecycle) and all 3 paradigms (Classical ML, Vector RAG, and LoRA/PEFT adaptation).
- **Uncapped Execution**: No limits on training epochs, dataset row counts, local model parameters, or HTTP inference requests.
- **Self-Contained**: CLI commands (`init`, `data validate`, `train`, `evaluate`, `serve`, `doctor`) operate entirely locally with zero telemetry locks or forced accounts.

---

## 2. B2B Enterprise Strategy: Direct Connectors & Custom Partnerships

Rather than locking down the core engine with restrictive licenses, commercial revenue is driven by high-value B2B enterprise needs:

```text
               ┌────────────────────────────────────────────────────────┐
               │              AIMLite Framework (Apache 2.0)            │
               │   100% Free: Full Core, CLI, RAG, LoRA, Classical ML   │
               └───────────────────────────┬────────────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
         ┌─────────────────────────────┐       ┌─────────────────────────────┐
         │  Enterprise Data Connectors │       │  Commercial AI Starter Kits │
         ├─────────────────────────────┤       ├─────────────────────────────┤
         │ • Snowflake / BigQuery      │       │ • Turnkey SaaS Churn Kit    │
         │ • PostgreSQL / Databricks   │       │ • Enterprise Support RAG    │
         │ • White-glove implementation│       │ • LoRA Fine-Tuning Kit      │
         │ • Dedicated B2B consulting  │       │ • Fast-launch boilerplates  │
         └─────────────────────────────┘       └─────────────────────────────┘
```

### Direct Outreach for Enterprise Connectors
Companies and data engineering teams store their actual production data in enterprise data warehouses (Snowflake, BigQuery, Databricks, PostgreSQL). 
- We partner directly with companies to build, maintain, and optimize custom data ingestion pipelines connecting their private data warehouses directly into AIMLite `Dataset` classes.
- Direct outreach and agreements allow flexible commercial agreements, tailored SLAs, and high-margin consulting contracts without altering the open-source core.

---

## 3. Alternative: The Fair-Code / Revenue Threshold Model

A modern alternative to pure AGPL dual-licensing is the **Fair-Code / Revenue Threshold model** (used by Unreal Engine, Sentry, and Ghost):

- **Free Tier Threshold**:
  - 100% free for any individual developer, educational institution, open-source project, or startup generating **under $100,000 in annual revenue** (or fewer than 10 employees).
- **Commercial Tier**:
  - Companies generating **more than $100,000 in annual revenue** must purchase an annual commercial license ($499 to $2,500/year).
- **Advantage**:
  - Completely eliminates friction for solo developers, students, and early startups.
  - Transparently bills funded startups and enterprises that extract commercial value from the framework.

---

## 4. Open-Core Enterprise Tier: Features Companies Pay For

In addition to licensing waivers, enterprises require specific compliance and data integrations that solo developers do not need. These capabilities form the **AIMLite Enterprise Extension**:

| Capability | Community Edition (Free) | Enterprise Tier (Paid) |
|---|---|---|
| **Core Framework** | Data, Model, and Lifecycle Pillars | Data, Model, and Lifecycle Pillars |
| **Paradigms** | Classical ML, RAG, LoRA Fine-Tuning | Classical ML, RAG, LoRA Fine-Tuning |
| **Data Sources** | Local files (CSV, JSON, Parquet, TSV, TXT) | Live data warehouse connectors (PostgreSQL, Snowflake, BigQuery, Databricks) |
| **Checkpoints & Logs** | Local directory persistence (`models/`, `checkpoints/`) | Centralized team checkpoint registry, cloud artifact synchronization |
| **Security & Auth** | Local development HTTP server | Role-based access control (RBAC), SSO/SAML, token authentication |
| **Compliance** | Standard logging | SOC2 / HIPAA compliant audit trails, model reproducibility lineage |
| **Support & SLA** | Community GitHub Issues / Discussions | Dedicated SLA, private Slack/Discord channel, priority bug fixes |
| **Pricing** | **$0 / Free Forever** | **$499 – $2,500 / year per organization** |

---

## 5. Immediate High-Margin Monetization: Turnkey Starter Kits

For immediate near-term revenue without enterprise sales cycles, AIMLite reference implementations can be packaged as commercial **Turnkey AI Starter Kits** (the "ShipFast for AI" playbook):

### 1. AIMLite SaaS Churn & Retention Kit ($149 single-license)
- Complete end-to-end churn prediction pipeline.
- Pre-processed dataset, trained classifier, inference API endpoints.
- Ready-to-use web retention dashboard that connects directly to customer databases and Stripe webhooks.

### 2. AIMLite Enterprise RAG Support Hub ($199 single-license)
- Turnkey knowledge base QA application.
- Automated document chunker, local/cloud vector store connector.
- Modern embeddable web chat widget with citation support.

### 3. AIMLite LoRA Instruction Specialist Kit ($249 single-license)
- Production-grade low-rank fine-tuning pipeline.
- Multi-adapter runtime hot-swapping pre-configured for customer support, ticket routing, and code assistant tasks.
- Parameter efficiency monitoring dashboard.

---

## 6. Implementation Roadmap

1. **Phase 1: Hackathon & Mindshare (Now)**
   - Launch `v0.1.2` on PyPI and GitHub with clear documentation and 49 passing tests.
   - Showcase the zero-path developer experience to win hackathon prizes and developer mindshare.
   - Establish AIMLite as the fastest way to build classical ML, RAG, and LoRA applications.

2. **Phase 2: Commercial Starter Kits (Months 1–3)**
   - Launch pre-built turnkey industry kits on Gumroad / LemonSqueezy.
   - Generate initial cash flow directly from developers and indie founders seeking to ship fast.

3. **Phase 3: Dual-Licensing & Enterprise (Months 3–6)**
   - Introduce the AGPLv3 / Commercial dual-license structure.
   - Offer the Enterprise tier (Snowflake/Postgres connectors + centralized team checkpoint sync) for organizations and corporate teams.
