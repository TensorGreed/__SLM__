---
sidebar_position: 12
title: API Surface Map
---

# API Surface Map

This is a practical map of core API areas.

## Project and Pipeline

- `GET /api/projects`
- `POST /api/projects`
- `GET /api/projects/{id}/pipeline/status`

## Domain Contracts

- `GET /api/domain-packs`
- `GET /api/domain-profiles`
- `POST /api/domain-blueprints/analyze`
- `POST /api/projects/{id}/domain-blueprints`
- `POST /api/projects/{id}/domain-blueprints/{version}/apply`
- `GET /api/projects/{id}/domain-runtime`

## Training

- `POST /api/projects/{id}/training/start`
- `GET /api/projects/{id}/training/model-selection/catalog`
- `POST /api/projects/{id}/training/model-selection/recommend`

## Evaluation

- `POST /api/projects/{id}/evaluation/run`
- `POST /api/projects/{id}/evaluation/llm-judge`
- `GET /api/projects/{id}/evaluation/gates/{experiment_id}`

## Export and Optimization

- `POST /api/projects/{id}/export/optimize`
- `GET /api/projects/{id}/export/deployment-targets`
- `POST /api/projects/{id}/export/{export_id}/run`

## Dynamic Catalogs

- `GET /api/targets/catalog`
- `GET /api/starter-packs/catalog`

## Runtime Introspection

- `GET /api/settings/runtime`
- `PUT /api/settings/runtime`

For full request/response schemas, use backend OpenAPI docs at `http://localhost:8000/docs`.
