---
title: Label taxonomy
sidebar_label: Label taxonomy
---

# Label taxonomy

A **label taxonomy** is the project's controlled vocabulary of labels. External datasets spell the same label many ways (`Leak 1`, `leakage`, `LEAK_1`); when a project has a taxonomy, the dataset-import mappers that carry labels (the [code-dataset mappers](./data-ingestion.md#findings-rows-code-datasets)) resolve every raw label to one taxonomy id, and rows with a label the taxonomy does not know are rejected as `unknown_label` instead of polluting the data. Each entry can also carry a [crosswalk](../reference/glossary.md#crosswalk) to reference ids in other namespaces.

Manage it on the **Data** page (Label taxonomy card, collapsed by default), or through the API below. It is stored with the project; no migration is needed.

## Entries

```yaml
name: toy
version: "2026.1"
entries:
  - id: LEAK-1
    title: Data leak
    aliases: [leakage, data exposure]
    crosswalk:
      stdA: [A-12]
      stdB: ["7.3"]
    severity: high          # unknown fields are kept in attrs
  - id: LEAK-1.1
    title: Leak in logs
    parent: LEAK-1
  - id: KEY-2
    title: Hard-coded key
```

Per entry: `id` (required, unique), `title` (defaults to the id), `parent` (an id, making a tree), `aliases`, `attrs` (any JSON) and `crosswalk` (`{namespace: [reference id, …]}`). Any other field is preserved in `attrs`. Ids and aliases are strings (whole numbers are converted). **Quote dotted numbers**: YAML reads an unquoted `1.10` as the number 1.1, so write `'1.10'`; a non-integer number as an id, alias, parent or reference is a `quote_number` error.

The parser accepts plain YAML / JSON only: YAML aliases (`*name`) are refused (`parse_error`), dates become ISO strings, sets become sorted lists, non-string keys become strings, and binary values (`!!binary`) are a `bad_attrs` error.

Accepted layouts (YAML or JSON): a top-level list of entries; a mapping with `entries` as a list **or** as a mapping keyed by id (the key is the id); or a mapping with exactly one other list / id-keyed collection (besides `name`, `version`, `description`). Top-level `name` and `version` become the taxonomy's `name` and `source_version`. Limits: 1 MB of text, 5000 entries.

### Crosswalk from fields

If your file lists reference ids as `"NAMESPACE:ref"` strings in some field, name that field in **crosswalk from fields** (`crosswalk_from` in the API). Each string is split at the first `:` into the crosswalk (`["stdA:12", "stdB:7.3"]` becomes `stdA: [12]`, `stdB: ["7.3"]`), duplicates are dropped, and the field leaves `attrs`. A value without a `:` (or `NAMESPACE:` with nothing after it) names a **whole namespace**: the entry gets that namespace with no reference ids, and validation warns `namespace_only_ref`. A value with no namespace (`:ref`) is a `bad_crosswalk_ref` error.

## Validation

Saving is all-or-nothing: any error rejects the whole taxonomy (422, `INVALID_TAXONOMY`, with every error in `metadata.errors`). **Validate** checks without saving.

| Code | Meaning |
|---|---|
| `parse_error`, `too_large`, `too_many_entries`, `bad_layout` | the text is not YAML / JSON, is over 1 MB / 5000 entries, or has no recognisable entry list |
| `empty` | no entries |
| `missing_id`, `duplicate_id` | an entry without an id; an id used twice |
| `unknown_parent` | `parent` is not an entry |
| `cycle` | parents form a loop; the message gives the path (`A -> C -> B -> A`) |
| `alias_conflict` | an alias equals another entry's id or alias, ignoring case and punctuation |
| `bad_crosswalk`, `bad_crosswalk_ref` | a namespace that is not a non-empty string, refs that are not a list of strings, or a `crosswalk_from` value with no namespace |
| `bad_attrs` | `attrs` is not a mapping, or a value is binary |
| `quote_number` | a non-integer number where an id, alias, parent or reference belongs; quote it (`'1.10'`) |

Warnings do not block saving: `no_title` (the id is shown instead), `ambiguous_id` (two ids look the same once punctuation is ignored; the exact id wins when resolving), `duplicate_title` (two entries share a title) and `namespace_only_ref`.

## Resolution

A raw label is looked up in tiers and the first tier with a hit decides:

1. the exact id;
2. an id or alias, ignoring case (punctuation kept);
3. an id or alias, ignoring case, whitespace and punctuation (`Leak 1`, `leak-1`, `LEAK_1` all find `LEAK-1`);
4. a title, ignoring case; then a title ignoring punctuation too.

If a key matches **more than one** entry within a tier it is **ambiguous** (for example `LEAK 1.1` when both `LEAK-11` and `LEAK-1.1` exist and the label is not an exact id): nothing resolves, the **Try a label** box lists the candidates, and the import mappers reject the row as `ambiguous_label` (the detail names the candidates). Use the exact id to disambiguate. The **Try a label** box on the card shows what a label resolves to and how it matched.

## Versions

The taxonomy's **version** is a 12-character hash of its content (independent of key order). Every save, restore and delete is remembered in a history of the last 20 versions, but only the newest 5 keep a full snapshot: **Restore** works on those (`restorable: true`); older entries stay listed with their version, entry count and date only, and restoring one is a 409 (the card disables the button). Import results and every findings row carry the taxonomy version they were resolved with, so you can tell which rows predate a change.

## API

| Request | What it does |
|---|---|
| `GET /api/projects/{id}/label-taxonomy` | the taxonomy, its `version`, `entry_count`, `namespaces` and `history` (`{version, entry_count, saved_at, current, restorable}`) |
| `PUT /api/projects/{id}/label-taxonomy` | body `{content` **or** `entries, format?, crosswalk_from?, name?}`; validates and stores |
| `POST /api/projects/{id}/label-taxonomy/validate` | same body; `{valid, errors, warnings, entry_count, version}`, nothing written |
| `POST /api/projects/{id}/label-taxonomy/restore` | `{version}`; 404 for an unknown version, 409 when its snapshot is no longer kept |
| `DELETE /api/projects/{id}/label-taxonomy` | removes it (history is kept) |
| `GET /api/projects/{id}/label-taxonomy/resolve?label=…` | `{raw, id, matched_by, title, candidates}` (`matched_by` is `id`, `alias`, `title` or `ambiguous`) |
