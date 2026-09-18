# AIMLite: Licensing Strategy & Commercialization Model

This document outlines the commercial strategy, dual-licensing structure, and sustainable monetization pathways for the AIMLite framework. It defines the boundary between the **forever-free Community Edition** and the **commercial enterprise tier**, balancing open-source viral adoption with high-margin enterprise revenue.

---

## 1. Core Philosophy: The Free Community Guarantee

To achieve grassroots adoption across developers, students, and startups, the core open-source framework must never feel artificially crippled or restricted.

### The Community Edition is:
- **100% Free Forever**: Available directly via `pip install aimlite` or `uv add aimlite`.
- **Fully Featured**: Complete access to all 3 core pillars (Data, Model, Lifecycle) and all 3 paradigms (Classical ML, Vector RAG, and LoRA/PEFT adaptation).
- **Uncapped Execution**: No limits on training epochs, dataset row counts, local model parameters, or HTTP inference requests.
- **Self-Contained**: CLI commands (`init`, `data validate`, `train`, `evaluate`, `serve`, `doctor`) operate entirely locally with zero telemetry locks or forced accounts.

> **Key Rule**: Individual developers, students, researchers, and early-stage builders will never be asked to pay for core framework capabilities.

---

## 2. The Dual-Licensing Model (The "Qt / MySQL" Architecture)

Inspired by the dual-licensing frameworks popularized by Qt, MySQL, and Neo4j, AIMLite operates under two parallel licenses:

```text
                     ┌───────────────────────────┐
                     │    AIMLite Source Code    │
                     └─────────────┬─────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
   ┌───────────────────────────┐       ┌───────────────────────────┐
   │    Open-Source License    │       │    Commercial License     │
   │      (AGPLv3 / LGPL)      │       │     (Proprietary OEM)     │
   ├───────────────────────────┤       ├───────────────────────────┤
   │ • 100% Free               │       │ • Paid ($1,000–$5,000/yr) │
   │ • Students & Hobbyists    │       │ • Closed-Source Startups  │
   │ • Open-Source Projects    │       │ • Corporations & Banks    │
   │ • Copyleft Obligation     │       │ • Zero Code Sharing Req.  │
   │   (share source if cloud) │       │ • Commercial Indemnity    │
   └───────────────────────────┘       └───────────────────────────┘
```

### How It Operates:
1. **The Open-Source License (AGPLv3 / Copyleft)**:
   - Anyone may read, fork, modify, and run AIMLite for free.
   - **Copyleft Clause**: If a commercial company modifies or embeds AIMLite within a closed-source SaaS platform or cloud backend, they are legally obligated to release the source code of their derivative software under the same license.
2. **The Commercial License (Proprietary Exemption)**:
   - Corporations, fintechs, and venture-backed startups cannot and will not open-source their proprietary business logic or closed models.
   - To legally waive the copyleft requirement, companies purchase an **AIMLite Commercial License**.
   - This grants the legal right to embed AIMLite in private, closed-source production applications without sharing their proprietary code, backed by commercial warranties and indemnity.

### Why Enterprise Legal Teams Willingly Pay:
Corporate legal departments have strict compliance policies prohibiting AGPL software in proprietary commercial offerings. When engineers adopt AIMLite, corporate legal will actively reach out to purchase a commercial license to satisfy their internal governance policies.

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
