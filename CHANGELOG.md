# Changelog

### Longer extra instructions for playbooks (October 29, 2026)

- The playbook `extra_instructions` parameter (scenario grid and code examples) now accepts up to 8,000 characters, up from 2,000. Limit change only: instructions of 2,000 characters or fewer produce exactly the same prompt as before, tags are still stripped, and longer text is still rejected with the same error naming the new limit.

### Code examples: format check, skips, extra instructions (October 29, 2026)

- A code example whose language is a data / markup format (JSON, YAML, TOML, XML) must now parse as that format, including when the label came from the source chunk (an XML chunk answered with Python and no declared language used to keep the label XML). Failures are rejected as `language_mismatch`. The teacher may also **skip** a chunk with nothing worth teaching: skips are reported (`skipped_seeds`, a few reasons), are not rejects, and a run where every chunk was skipped fails with a message naming the reasons. New `extra_instructions` parameter (up to 2,000 characters) adds your own guidance to the prompt.
- **Check code formats** in the Generate step's review queue re-checks pending code examples: it shows how many do not parse as their label, grouped by label and kind of failure; you pick the groups, and only those rows are marked rejected (`language_mismatch`, never deleted; restore them from the rejected view's Restore group once it ships). Template expressions in YAML, multi-document YAML and XML fragments with several sibling elements are accepted. API: `POST .../synthetic/review-queue/recheck-code-format` with optional `dry_run` (default true) and `row_ids`.

### Seed filters for code and document playbooks (October 29, 2026)

- Playbooks that seed from repositories or documents can now narrow the seed pool before anything is sampled. **Seed filters** in the playbook picker (and `seed_path_include`, `seed_path_exclude`, `seed_relevance_terms`, `seed_relevance_scope`, `seed_relevance_match`, `relevance_exempt_source_ids` on `POST .../run-playbook`) keep passages by path glob and by whether a term appears in the passage's path or text (or, in file scope, anywhere in its file); chosen sources can be exempt from the term filter. `seed_relevance_match: "text"` matches terms only in passage text, so generic files in topically named folders no longer pass on their path. Filters run before the 2,000-passage cap, so a big repository's off-topic files no longer crowd the sample.
- The result's `seed_pool` (and the dry run) reports, per source, how many passages there were, how many were kept, and how many each filter dropped; the bell shows the totals. Filters that leave nothing fail the run with those counts instead of finishing empty. Invalid globs, over-long lists or a bad scope are rejected with `400 invalid_playbook_param`.

### Plan rules: your own consistency checks for structured plans (October 29, 2026)

- A structured-plan project can now carry a **rules document** (JSON) that BrewSLM checks deterministically, with no model: a step must satisfy a condition (`step_requires`), a step that refers to an object another step created needs that creating step to satisfy a condition, and a reference nothing creates can itself be a violation (`ref_requires`), one step shape needs another (`requires_step`), two shapes must not appear together (`forbid_together`). Conditions look at dotted paths inside a step (list wildcards allowed) with equals / in / exists / regex / max / min and friends; operations are matched by tool + `args.operation` (both paths configurable). Each rule has an id, a description and a severity (`reject` or `warn`). A malformed document is a clear 400 (`invalid_plan_rules`) naming the bad part.
- At generation, pass the document as `params.plan_rules` to the `scenario_grid` playbook: after schema validation, a plan breaking a `reject` rule is dropped as `rule_violation` (the Job's reject summary lists the rule ids); `warn` rules only note the rule ids on the accepted row.
- On rows already in the review queue, **Check plan rules** (structured-plan projects) takes pasted or uploaded rules, does a dry run (how many pending plans break each rule, with example rows), and after you confirm marks them rejected as `rule_violation:<rule id>`. They stay on disk grouped by rule in the Rejected section and are never purged. Every group in the Rejected section now has **Restore group** (any reason, not only plan rules): the rows go back to the pending queue with their reject reason cleared (`POST .../synthetic/review-queue/restore`, dry run by default). Rules check missing paths strictly (optional `missing: "pass"`), order steps by list position, flatten list-valued references, and cap regex size. API: `POST /api/projects/{id}/synthetic/check-plan-rules` (`dry_run` defaults to true; `?async_job=true` runs the `plan_rules_check` Job).

### Review assist: several judges per row (October 29, 2026)

- Running the pre-screen with a second model no longer overwrites the first model's verdicts: each row keeps **one verdict per model**, and running the same model again replaces only that model's. The queue shows where the models stand (**agree pass**, **agree fail**, **disagree**, **single**, **none**), can filter and group by it (**Judges agree?**) and show one model's verdicts (**Judge** filter); every row card lists each model's verdict and note, and group labels name the models. **Accept group** is offered only where every model that ran says pass (two models agreeing, or the one model that ran); disagreements are never bulk-accepted, and a row another model disagrees on is skipped even in a plain Verdict group. A verdict only counts while the row is unchanged.
- The pre-screen can resume (**Only rows this model has not judged yet**) and aim a second opinion at the rows that need it (where another model said fail or unsure, or where the models disagree). Its result and the bell add "agree N · disagree N" once two models have judged the same rows. API: `only_missing_judge`, `consensus`, `prior_verdicts` on `POST .../review-assist`; `?assist_consensus=` / `?judge=` and `group_by=assist_consensus` on the review queue. Existing single-verdict files are read as that model's verdict and converted on the next write, nothing lost.

### Review assist: whole plans, adaptive calls, temperature 0 (October 29, 2026)

- The pre-screen now shows the model the artefact it is judging **whole** (the full plan, code example or answer, after the task) and only shortens the cited passages (each marked "[truncated]"; the model is told so). Before, long structured-plan rows were cut at 6,000 characters, so the model often never saw the verification or rollback. A row whose task and answer alone are too long to read whole is marked `unsure` / `too_long_to_judge` instead of being judged on a cut. Calls hold fewer rows when the rows are big, to stay inside a local model's 32k context. Judging runs at temperature 0, conflicting duplicate answers become `unsure` / `judge_conflict`, and verdicts are written in a handful of file writes instead of one per call.
- **Accept group** / **Reject group** re-read the group right before acting, and Accept is no longer offered on "unsure" or "(not pre-screened)" groups. Rows record the backend without any credential embedded in its URL; documents-to-Q&A rows now record their Job and backend; a duplicated project no longer points its rows at the original's Job.

### Review assist: pre-screen pending synthetic rows with a model (October 29, 2026)

- A new **Pre-screen with a model** action in the Generate step's review queue sends the pending synthetic rows to a model of your choice (a local model, the local CLI, or a cloud model; Anthropic cloud models can run as a half-price Message Batch). Each row gets a verdict (**pass**, **fail** or **unsure**), a reason code and a one-sentence note. Built-in rubrics cover code examples (correctness, use of the cited passages, unsafe patterns such as hard-coded secrets, disabled TLS verification, logged sensitive values and string-built SQL), structured plans (steps consistent with the scenario and any allowed operations you type in, meaningful verification and rollback, no invented fields) and question-and-answer rows (supported by the passages); other rows get a generic rubric, and your own text is appended to it. Rows are treated as data: the model is told to ignore instructions inside them, and an answer it did not give or that cannot be read is `unsure`, never a pass.
- The pre-screen never accepts or rejects anything. Filter or group the queue by verdict or by verdict and reason, then **Accept group** or **Reject group** (rejects are tagged `assist:<reason>` so the Rejected section groups them by why they failed). The bell shows "pass N · fail N · unsure N" and the spend; an API budget stop keeps the verdicts already written. Verdicts are kept in a side file next to `synthetic.jsonl` so the pre-screen never rewrites your rows; running it again replaces the earlier verdicts. API: `POST /api/projects/{id}/synthetic/review-assist?async_job=true`.

### Synthetic row provenance (October 29, 2026)

- Every row a playbook writes now records which Job and backend produced it: `synth_job_id` (null when run synchronously), `synth_backend` (the backend pin, e.g. `cloud:anthropic:<model>` or `command:<model>`) and `synth_batch: true` for a Message Batch run. Legacy synthetic tasks stamp the Job and model too.
- The review queue returns these on each row, can group by **Job** or **Backend** (group labels read "job:12 · backend · mode · batch/direct" so you can match them to the bell) and filter to one Job or one backend (`?group_by=job|backend`, `?synth_job_id=`, `?synth_backend=`). Rows written before this change show as "(no job)" / "(unknown backend)".

### Code examples: declared answer language (October 28, 2026)

- The `code_examples` playbook asks the teacher to state the language of each answer and parse-checks the code under it, so a Python answer to a JSON / YAML / config chunk is no longer rejected as `code_does_not_parse` (the check used the chunk's language). A declaration for a language with no runnable check, or an unknown one, falls back to the chunk's language, so it cannot skip a check; a code chunk can never be re-labelled YAML (that switch needs a data / config chunk and a YAML mapping or list reply). Rows keep the final `language` and, when different, `seed_language`.

### Pipeline evaluation (October 27, 2026)

- **Pipelines**: chain trained projects into stages (extract → rewrite, classify → answer, detect → fix) on a new **Pipelines** page. Each stage picks a project and model (served, base or a run), a recipe prompt or a `{field}` template, inputs from the held-out row, the previous stage's output or the current fan-out item (JSON pointers), optional fan-out over a list with item-key matching, retrieval, max tokens and JSON / text output. Saving reports every invalid field at once. Stored as files under `DATA_DIR/pipelines/` (no migration).
- **Held-out set with gold per stage**: JSONL rows `{id, input, gold: {<stage>: ...}}` with gold for any subset of stages; invalid rows are rejected by reason with line numbers and the valid ones kept.
- **Two views per stage**: end to end (from the previous stage's prediction) and isolated (from its gold), each scored by the stage project's own evaluator; fan-out items are matched by key, with spurious outputs and missed items counted.
- **Where it breaks**: the share of final items that are fully correct and, for the rest, the first stage that went wrong (a missed item counts against the stage that failed to produce it), with project, model and run named on every stage. Runs are background jobs (`pipeline_eval`) capped at 5000 model calls; a run that produced or scored nothing fails with the reason. Model calls use the serving gateway's internal path, which neither uses a key's rate limit nor writes request logs. See [Pipeline evaluation](slm-docs/docs/workflows/pipeline-eval.md).
- **Access**: with authentication on, a pipeline is visible only with viewer access to every member project and editable / runnable only with editor access to all of them. Each member project's Results page links to its pipelines.
- **Stage scores you can trust**: each stage output is scored the way its project's held-out eval scores it. The output becomes a row in the project's own field names (stage inputs, the stage prompt, the gold under the prepared rows' reference fields), built by the project's prompt builder and carrying the same per-row extras held-out eval uses (the code and finding behind a patch, a plan's scenario, gold tool calls, ...), so recipe scorers no longer pass outputs they never checked. The judges held-out would run (long-answer judge, structured-plan rubric, remediation judge) run per stage view with spend metered to the stage's project. An output is **unscored with a reason** instead of right or wrong when the scoring row cannot be built (`row_context_unavailable`), a judge would decide it but none ran (`needs_judge`, stage `judge_not_run`) or the scorer failed (`scoring_error`); such outputs and the final items that pass through them are left out of the accuracy, the end-to-end share and the attribution, and the stage card says why.
- **Per-item attribution**: the stage that produces the fan-out items is right for an item only when it produced that item as in gold (strings compared like exact match), not merely when its key matched; the stage card shows how items were compared. One wrong item no longer marks every item of the row as failing upstream.
- **Hardening**: item keys with a missing value match nothing (`missing_item_key`; such gold items are rejected on upload), duplicate keys are counted (`duplicate_item_key`, first wins), items lost because the upstream output failed are `lost_to_error`, the isolated view runs only where it can be scored, unknown `{placeholders}` are a save error, a recipe builder crash is a `prompt_error`, stage models resolve by the project's real name, each run re-checks the held-out gold against the current pipeline (`gold_mismatch`), uploads are capped at 50 MB (413), cancelling checks before every model call, a truncated run names the row it stopped at, the in-request run path takes at most 5 rows (else use a job), and runs whose job died show as `stale` (the runs list reads small summary files). The page shows running / failed runs as a status panel, run status in the lists, a "project missing" badge, isolated skipped rows, and no longer turns a max-tokens value of 0 into 512.
- **One-line code fences**: a reply whose closing fence sits on the last content line (```` ```json\n{"a": 1}``` ````) now parses in the gateway's guided-JSON check, the structured-output evaluator and pipeline stages (arrays too). Held-out scores of existing structured-output projects can change slightly where models answered this way.
- **Fix: tool-call held-out eval never received gold calls (since the tool-catalog release) — re-run tool-call evals.** Real inference builds fresh prediction rows, and the gold calls the `tool_calling` scorer compares against were not copied onto them, so every tool-call held-out eval scored 0 whatever the model said (tests copied the whole row and hid it). Held-out and pipeline evaluation now share one helper for per-row scoring extras.

### Multi-turn chat training, per-turn eval and preference pairs from eval (October 27, 2026)

- **Assistant-only loss for chat rows**: a messages / conversations row with a system message, more than one assistant turn or more than two messages is rendered with the tokenizer's chat template and trained on assistant tokens only (spans found by prefix rendering; each assistant turn's end-of-turn token is trained, system / user / tool tokens are masked; messages after the last assistant turn are trimmed). Templates whose per-turn renders do not line up fall back to the final assistant turn only (counted as `mask_fallback_rows`). A tokenizer without a template renders `role: content` lines. Rows trained on every token before; a `[system, user, assistant]` row is now masked too, while a plain `[user, assistant]` row trains byte-identically.
- **Truncation keeps the dialogue sound**: an over-long row drops its oldest exchanges whole (never mid-turn), keeps the system message and the final exchange, and is dropped (and counted) if the final exchange does not fit.
- **Run report**: `multi_turn` (at experiment `config._runtime.multi_turn`) with `rows`, `assistant_token_share`, `assistant_turns`, `truncated_rows`, `dropped_turns`, `dropped_rows`, `mask_fallback_rows`; a simulated run reports the same block marked `approximate` (about 4 characters per token). `runtime_environment.loss_masking` is `assistant_only` or `mixed:completion_only+assistant_only`.
- **ORPO fix**: ORPO mode falls back to `trl.experimental.orpo` for `ORPOTrainer` / `ORPOConfig` when the installed trl no longer exports them at the top level (it raised before).
- **Last-turn eval**: a held-out chat row is scored on its final assistant turn with the whole history before it (system prompt included) rendered by the chat template, like training. Single-turn rows are unchanged.
- **Judge every assistant turn**: `dialogue_judge: true` on `run-heldout` replays each assistant turn of multi-turn rows from the gold history and grades it with the answer judge; `metrics.dialogue_judge` reports `mean`, `worst_turn_mean` and `by_turn_index`. Off by default; skipped without a reachable judge.
- **Judged rows are stored in a side file**: `details.row_predictions_file` (`{path, count, truncated}`) points at `experiments/<run>/evals/result_<id>_rows.jsonl`, holding only the rows a judge scored (history capped at the last 40 messages, no duplicate prompt, at most 1000 rows); the results list never carries row arrays.
- **Chat judge**: `chat_sft` is a judged profile (the judge sees the whole dialogue as the question), and pairs fall back to the per-turn judge's final-turn verdict for chat rows the answer judge did not grade. Rows past the per-turn replay cap are left out and reported (`rows_not_replayed`, `replay_cap_hit`), and replayed generations no longer count in latency / throughput stats.
- **Chat-template edge cases**: templates that append the end-of-sequence token after the last message (Phi-3 style) keep every assistant turn trained; no-template multi-turn rows get a trained EOS after the final turn; Qwen3 think-block and gpt-oss channel templates are known final-turn-only families; the `trl_sft` backend still trains multi-turn rows on every token.
- **`documents_multi_turn` playbook** (`generic-sft`): passage-grounded dialogues from the project's documents with follow-up questions and `[n]` citations in every assistant turn (`turns_min`, `turns_max`, `dialogues_per_passage_group`, `passages_per_dialogue`); unsupported, uncited, repeated, mis-ordered or test-question dialogues (`test_question_leak`) are rejected by reason and bulk-droppable in the review queue. Each dialogue starts with a system message holding its passages in the serving preamble, so the `[n]` citations resolve at train time.
- **Preference pairs from an eval**: `POST /api/projects/{id}/alignment/preference-pairs/from-eval` turns the answer judge's wrong / partial rows into `{prompt, chosen: answer key, rejected: model answer}` pairs, with skipped rows counted by reason; nothing built is a 400 with the breakdown. Pairs from an eval on the `test` split are refused (`test_split_refused`, memorisation) unless `allow_test_split` is set; every pair carries `source_split`. See [Multi-turn chat](slm-docs/docs/workflows/multi-turn-chat.md).

### Serving gateway, multi-LoRA plan and MCP server (October 27, 2026)

- **OpenAI-compatible gateway**: `/v1/models` and `/v1/chat/completions` (plus `stream: true`) answer with a project's trained runs, addressed as `<project-slug>`, `<slug>:run-<id>` or `<slug>:base`. Keys are scoped to projects with a per-minute rate limit (`/api/gateway/keys`; shown once, stored hashed); `/v1` requires a key even when `AUTH_ENABLED=false`. A project can run in process or forward to a vLLM / Ollama / OpenAI-style server.
- **Retrieval with citations**: runs trained with passages retrieve by default (`brewslm: {retrieval, k}` overrides); the preamble is exactly the training one and answers carry `citations` with `source_url`. A missing index skips retrieval instead of failing.
- **Guided JSON**: the project's JSON schema is sent as a strict `response_format` json_schema to every upstream kind (Ollama's `/v1` ignores `format`; recent vLLM dropped `guided_json`) and the reply is validated and reported (`valid`, `errors`); in process it is validated only. A client-sent `response_format` wins. Upstream failures are a `502` without the upstream body.
- **Hardening**: deleting a project removes it from every gateway key's scope (emptied keys are revoked) and deletes its request log; keys are listable and revocable by their creator, an admin or the owner of every scoped project; `/v1` errors (404, 405, 422, 500) keep the OpenAI envelope; the MCP HTTP mode checks Origin, Host and Content-Type and refuses a non-loopback host without `--allow-remote`; multi-LoRA plans reject adapters above rank 256 or without weights.
- **Named tools and MCP server**: saved prompt tools per project (`{field}` templates, input schema, `@recipe` for the code-findings prompt) and `python -m app.mcp_server` (stdio, optional HTTP) exposing `list_models`, `run_model`, `retrieve` and the named tools to Claude Code. See [Serving gateway and MCP server](slm-docs/docs/deployment/gateway-and-mcp.md).
- **Multi-LoRA plan**: `POST /api/projects/{id}/serve/multi-lora-plan` builds one vLLM launch (`--enable-lora`, `--lora-modules`) for runs sharing a base model; `.../multi-lora-runs/start` launches it.
- **Telemetry**: per-request latency, status and token counts per model (`GET /api/projects/{id}/gateway/telemetry`); prompt and reply text are never stored. Migration `20261027_0051`.

### Tool catalog (MCP / OpenAPI) and tool-call scoring (October 27, 2026)

- **Hardening**: a stored secret is reused only for the same kind and endpoint; user env names that change how a binary loads code are refused; unknown destructiveness fails closed; `tool-call-json` now trains on the canonical call JSON (`{"tool": ..., "args": ...}`) instead of the tool result, matching what `tool_calling` scoring reads; cached `tool_calling` base results are reused only under the same tool-call scoring fingerprint.

- **Tool catalog**: a project keeps a versioned list of the tools a model may call, imported from an MCP server (stdio or streamable HTTP), an OpenAPI document, entered by hand or shipped in a domain pack. Per-tool allowed / read-only flags (a destructive tool can never be made read-only), history with restore, secrets kept only in the project secret store. Stdio launches a configured process, so it is off until an administrator allow-lists the binary (`MCP_STDIO_ALLOWED_BINARIES`). See [Tool catalog](slm-docs/docs/extensions/tool-catalog.md).
- **Tool-call scoring**: `tool_call_valid` (known, allowed tool, arguments valid against its schema), `tool_args_match` (against the answer key, typed) and `plan_order_valid` (unique ids, existing dependencies, no cycles). A `structured-plan` row's tool verdict folds into its pass (`metrics.tools`, `tool_call_valid_rate`, `plan_order_valid_rate`); the prompt lists the allowed tools, identical for training and eval, and the scoring fingerprint follows the catalog version and flags. Settings API `GET` / `PUT /api/projects/{id}/tool-scoring-config`.
- **`tool_calling` profile**: a task handler scoring one call per row, headline `tool_call_pass_rate` (also `tool_selection_accuracy`), paired with the base model; the `tool-call-json` adapter prepends the same tool block when a catalog exists. Projects without a catalog keep their old prompts and scores.
- **Dry run** (admin switch `MCP_DRY_RUN_ENABLED`, off by default): valid read-only calls can be executed once against the live MCP server after scoring (informational; at most 50 per eval). Mutating tools are never executed.

### Structured-plan recipe, JSON-Schema bundles and a rubric judge (October 27, 2026)

- Added the **Structured plan** recipe (`structured-plan`, structured extraction, scoring mode `structured_plan`): a scenario (and optional reference passages) in, one JSON plan out that must validate against a JSON Schema bundle you upload (several files joined by `$ref`, resolved inside the bundle only, no network). Suggested base `Qwen/Qwen2.5-7B-Instruct`.
- **Scoring**: valid JSON, schema validity (grouped top errors), and JSON-pointer field matches (wildcards, order-insensitive multisets, several acceptable values per pointer). Headline `structured_plan_pass_rate` paired with the base model; base results are reused only under the same scoring fingerprint.
- **Rubric judge**: an optional YAML / JSON rubric of weighted criteria with five anchors each and critical flags, graded by the same judge the answer judge would pick (local model first); per-criterion table, `rubric_pass_rate` and `rubric_mean_score`. Without a judge the rubric is reported as unavailable and the pass ignores it.
- **Settings API** `GET` / `PUT /api/projects/{id}/structured-plan-config` and `POST /api/projects/{id}/structured-plan/validate`; a valid schema also becomes the project's structured-output hint (`guided-json`) and the export manifest carries it.
- **Playbook `scenario_grid`**: write scenarios from a matrix of dimensions (uploaded markdown table or CSV), ground them in the project's documents, and have the teacher write plans that are validated against the schema before they are accepted.
- UI: a "Structured plan" card on Train (schema upload, rubric, pointers, toggles, paste-a-JSON validator), the structured-plan metrics panel and row view in Evaluate, and the scenario-grid parameter form.

### Code-remediation recipe + code-check registry (October 26, 2026)

- Added the **Code remediation** recipe (`code-remediation`, structured extraction, scoring mode `remediation`): a code chunk (absolute line numbers) plus one finding plus optional reference passages in, `{patch, rationale, citations, ...declared extra fields}` out, the patch a unified diff against the shown lines. One shared prompt builder serves training and held-out eval (chat template, byte-identical). On a project with a document index, passages are retrieved once at dataset prep (never the row's own chunk) and stored on the prepared row, so eval sees the same prompt. Settings: `GET` / `PUT /api/projects/{id}/remediation-config` (detector, checks, judge, passages, top_k).
- **Mechanical scoring**: `patch_applies` (a strict pure-Python unified-diff applier: exact context, bounded offset search, ambiguous / overlapping / out-of-range hunks refused, the body wins over header counts), `code_parses` on the patched code (only rows whose original code passes the checks count, so base and fine-tuned share denominators), `issue_resolved` (a configured detector — a code-findings project's trained run, or an allow-listed external command — runs on the original and the patched code; the label must be gone and no new labels appear; rows the detector can't see are excluded), `citations_valid`, extra-field accuracy and an optional judge with a remediation instruction (the default judge prompt is unchanged). Headline `remediation_pass_rate`, paired row by row against the base model; a cached base result is reused only when the scoring settings match.
- **Code-check registry** (`GET /api/code-checks`, `POST /api/code-checks/run`, plugins via `CODE_CHECK_PLUGIN_MODULES`): built-in Python, JSON, YAML and JSON-Schema checks; external Java, Go, Terraform, shell, JavaScript and C# checks, each only when allow-listed in `CODE_CHECK_ALLOWED` and installed. Fragments are wrapped and only syntax errors count. External checks run sandboxed: temp dir, no shell, scrubbed environment, CPU / memory / file-size limits, timeout, network isolation when the host allows it (`CODE_CHECK_NETWORK_ISOLATION=require` to insist).
- **Synthetic playbooks** `remediation_from_findings` (from findings rows, fix pairs and answer-key dev rows, never the test set; fix pairs keep their real diff) and `code_examples` (task → idiomatic code from code-document chunks, cited as passages), with grouped reject reasons (`patch_does_not_apply`, `code_does_not_parse`, `bad_citation`, `copied_chunk`, ...).
- UI: remediation metrics panel and row view (patch as a diff, status chips, cited passages) in Evaluate, a "Remediation checks" card on Train, playbook parameter forms.

### Domain packs as data + your own checks (October 26, 2026)

- A **domain pack** can now carry content as well as its overlay and hooks: a zip with `pack.yaml` plus a label taxonomy, probe packs (your own checks), rubrics, schemas, a tool catalog, seed documents (licence + usage each), answer-key dev / test rows and a project template (recipe, base model, target and plan profile, training defaults, retrieval settings). Domain content ships as data, never as code; no pack ships in core.
- **Import** (`POST /api/domain-packs/import/preview`, `/import`): validated before anything is written (missing or unsafe paths, zip-slip, size limits, corrupt zips, duplicate ids, bad regexes, unknown recipe, invalid taxonomy / schema / contract; `INVALID_DOMAIN_PACK` with `metadata.errors`, grouped by code in the UI). A re-import shows a per-section diff; identical content is `unchanged`, otherwise a version that isn't higher is patch-bumped. Each version's files are kept at `data/domain_packs/<pack_id>/<version>/`; system packs can't be imported over. No migration.
- **Create a project from a pack** (`POST /api/domain-packs/{pack_id}/instantiate`): copies every artifact, all-or-nothing, stamped with `pack_id@version` (documents get a `domain_pack` provenance source). **Export a project as a pack** (`GET /api/projects/{id}/domain-pack/export`); the zip re-validates and round-trips. Preview, import and instantiate need the admin or engineer role. Training defaults prefill the Train form; they are not applied to API or autopilot starts.
- **Your own checks** (user probe packs, `GET` / `PUT` / `DELETE /api/projects/{id}/probe-pack/user`, `/validate`, `/restore`): any existing probe property plus `must_match` / `must_not_match` (regex, 0.5 s timeout per search) and `json_pointer_not_in` (JSON pointer with `*` wildcards, every JSON value in the output is checked), with `severity` (weight) and `critical`. They run beside the built-in checks in the same evaluation; `probe_pass_rate` stays built-in only, your checks report `user_probe_pass_rate` and `by_source`. **Any critical check that fails blocks the gate** (`probe_critical_failures`, required, always on even when the probe gate is off; it also fails when the checks didn't run).
- UI: Settings > Domain packs gains import with preview and diff, the pack list with "Create project from this pack" and "Export this project as a pack"; the checks panel lists your checks (source, severity, critical), critical failures and per-source pass rates, with a paste-YAML editor (merge by pack id, restore an earlier version); the scorecard groups the critical gate under "Independent ruler".

### Code-findings recipe + finding-set scoring (October 26, 2026)

- Added the **Code findings** recipe (`code-findings`, structured extraction, scoring mode `finding_set`): a chunk of code in, `{"findings": [{label, start_line, end_line, evidence, attrs}]}` out, with absolute line numbers; suggested base `Qwen/Qwen2.5-Coder-7B-Instruct`. One shared prompt builder serves training rows, held-out eval and probes (byte-identical, chat template, every line numbered, taxonomy labels listed); rows are detected from the G3 importer shape. The structured adapter only uses it when the project's scoring mode is `finding_set`; a findings-less row is rejected as `findings_json_missing`. The prepared manifest stamps `finding_match`, `labels`, `label_parents` and `prompt_labels`.
- **`finding_set` metrics**: strict label + line-range IoU matching (default 0.5, one-to-one), micro and macro precision / recall / F1, a per-label table, `clean_chunk_fp_rate`, `malformed_rate`, `json_validity_rate`, attribute accuracy and an optional hierarchical score (parent-label credit). Scaffolded pass/fail rules use `lte` for the `max_` rates. Empty-on-empty scores 1.0 for macro as for micro.
- Three **synthetic playbooks** (`seeded_defect`, `hard_negatives`, `cluster_targeted`) write findings rows and check the teacher's claim against the code it returns (changed-line anchoring, real evidence, lines inside the chunk). Seeds come from the answer key's dev set and the project's code documents only (never the test set, never evaluation-only / retrieval-only sources) and rotate across batches; a context dataset cannot be an answer key (`context_dataset_not_allowed`). Every playbook run now reports `rejected_count`, `reject_summary` and `rejected_examples`, kept across resumed chunks. `params.label_id` is a free label when the project has no taxonomy.
- **Structured-output hint**: applying the recipe stores a JSON Schema (label enum from the taxonomy) at `GET /api/projects/{id}/guided-json`, refreshed when the taxonomy changes and carried into the export manifest.
- A recipe-keyed **probe pack** (empty / comment-only code reports nothing; labels stay put when a reassuring or instruction-like comment is added or variables are renamed), run with the full JSON generation budget.
- UI: finding-set metrics panel and row view in Evaluate, a seeded-defect form (label, attributes, data-carrying context with a dataset and column picker) and a reject summary on the synthetic playbook panel, a guided-JSON card on Ship > Package, rejected counts in the notification bell. New `GET /api/projects/{id}/datasets` and `GET /api/projects/{id}/datasets/{dataset_id}/columns`.
- Training preflight no longer reads a code row's `path` as a media file; the synthetic dedup keys findings rows on code plus findings.

### Label taxonomy + code-dataset mappers (October 26, 2026)

- **Label taxonomy** per project (`runtime_config["label_taxonomy"]`, no migration): entries with id, title, parent, aliases, attrs and a crosswalk to reference ids in other namespaces; YAML / JSON in several layouts, `crosswalk_from` for `NAMESPACE:ref` fields, all-or-nothing validation (`INVALID_TAXONOMY`, per-entry `metadata.errors`), tiered resolution by id / alias / title (exact first; a label that matches several entries is ambiguous and never guessed), YAML aliases refused, a content-hash version and a 20-version history with restore (full snapshots for the newest 5, older entries listed but not restorable). A `crosswalk_from` value without `:` names a whole namespace. New `GET` / `PUT` / `DELETE /api/projects/{id}/label-taxonomy`, `/validate`, `/restore`, `/resolve`.
- **Code-dataset mappers** writing canonical findings rows (`row_kind: "findings"`: code, language, path, absolute finding lines, taxonomy-resolved labels, `taxonomy_version`): `code_label_to_findings` (labelled code), `fix_pair` (before / after pairs to findings, patch or preference rows; finding lines from a line diff) and `contrast_functions` (files with `bad…` / `good…` functions, via the code chunker). `fix_pair` finding lines are per changed hunk by default; `contrast_functions` anonymises `bad…` / `good…` names and blanks revealing comments by default so a model cannot learn the names. Unknown / ambiguous labels reject as `unknown_label` / `ambiguous_label` and can be bulk-dropped; an odd row is rejected as `parse_error` instead of stopping the import; without a taxonomy labels stay as written with a `no_taxonomy` warning. Dataset-import preview / run results gain `taxonomy: {version, entry_count}`; identical per-row warnings are listed once.
- The dataset-import introspector recognises code columns and proposes the three mappers (and ranks plain text classification below them for code data).

### Command-line teacher backend + hard API budget (October 26, 2026)

- Added the **command (local CLI) synth backend** (`command` / `command:<model>`): runs an allow-listed command-line tool (for example a headless vendor CLI on a subscription) as the teacher. Explicit pin only (never auto-picked, not even as the judge); disabled until `SYNTH_COMMAND_ALLOWED_BINARIES` is set. No shell, minimal environment (API-key variables never forwarded), a temporary working directory per call, killed on timeout / cancel; argv, binary, environment and prompt mode are install-wide, never per project. Requests are packed into JSON-lines batches (`SYNTH_COMMAND_BATCH_SIZE`) with per-item retries and reasons; the documents-to-Q&A flow, teacher answers and relabel use the packed path, and the flow's **Batch** option now accepts it.
- A usage-limit failure **pauses and resumes**: the Job message reads "waiting for usage window - resumes ~HH:MM UTC" (a "paused" chip in the bell), the same call is retried, and the Job result records `paused_seconds`. Synthetic playbooks run **chunked and checkpointed** on this backend (`SYNTH_COMMAND_PLAYBOOK_ROWS_PER_CALL`): a re-run reuses finished chunks, failed chunks are listed in `failed_chunks`.
- **Hard API budget** for metered cloud calls: the per-project monthly budget is now a hard stop, and `LLM_MONTHLY_BUDGET_USD` adds an account-wide cap across all projects (editable in runtime settings). A job pinned to a cloud model starts as `402 llm_budget_exceeded` at the cap; a job that reaches it mid-run fails with "API budget reached (...)". Local models and subscription CLIs are never counted or blocked. `GET /projects/{id}/llm-budget` gains `global` and `hard_stop`.
- Spend: subscription CLI usage is recorded as `subscription:<binary>`, priced $0 and shown separately (`billing` per model row; `subscription_calls` / `subscription_tokens` in the totals).
- Hardening: a `command:<model>` pin may not start with `-`; usage-limit pauses have a 60 s floor and one call gives up after 20 consecutive hits; a bare allow-list name only matches a bare `argv[0]`; cancelling a Job kills the in-flight CLI and sibling chunks; concurrency is one process-wide limit (a project override can only lower it); Message-Batch and unpriced-model spend count toward the hard budget; the playground, probe judge, plan refinement and teacher capture stop at a reached budget; playbook checkpoints are keyed on the prompt and deleted only after the rows are committed.
- New `GET` / `PUT /api/projects/{id}/eval-judge` (the judge model picker, with billing per option); `GET /projects/{id}/synthetic/backends` entries gain `billing` and `detail`.

### Licence & provenance ledger (October 26, 2026)

- Added a **licence & provenance ledger**: one source per upload, website crawl, git repository, Hugging Face / Kaggle / URL import, dataset-import run, predefined split, demo and external benchmark (`GET` / `PATCH /api/projects/{id}/provenance/sources`; migration `20261026_0050`). Each records where the data came from (URL / dataset id / repo, commit or revision), when, and its licence (read from the hub / repository where published; unknown otherwise, with warnings for unknown and non-commercial licences; a licence can be marked verified).
- Added per-source **usage** (`any`, `train`, `eval`, `rag`): `eval` rows are held out of train / validation and go to the test split (duplicates of them - near-identical rows too when dedup is on - are removed from train and validation; not added to a pinned test), `rag` rows are in no split and never seed synthetic data (their documents stay indexed for passages). The prepared manifest gains `provenance` (rows per source per split, exclusions, licence warnings) and training runs record it (`data_provenance`); projects with no eval / rag source prepare exactly as before.
- Exports carry `provenance` (sources, rows per licence, warnings) in `manifest.json` and a "Data sources & licences" section in `RELEASE_NOTES.md`. Uploads accept `usage` and `license_info` (batch too); the website crawl accepts `licence`; `DocumentResponse` gains `data_source_id`, `usage`, `licence`, `licence_flags`, `source_ref`, `retrieved_at`; the `Imported from ...` placeholder licence is no longer written.
- A duplicated project remaps `source_document_id` in the copied cleaned / synthetic / prepared rows, so eval / retrieval-only marks keep working in the copy; rows generated from cleaned chunks are stamped with their documents; answer-key (gold) building is grounded in eval-only documents too; sources left with no documents are removed; more non-commercial wordings are flagged (`none` / `n/a` count as unknown).

### Document sources + code-aware chunking (October 26, 2026)

- Added **Documents from website / git repository**: `POST /api/projects/{id}/document-sources/web-crawl` and `/git-repo` (background Jobs `document_source_web_crawl` / `document_source_git_repo`). The crawler honours robots.txt, stays inside allowed prefixes, rate-limits, caches by ETag / Last-Modified and strips navigation; the git importer pins the commit, detects the licence and skips binaries, large files and vendored directories. The website importer follows RFC 9309 (longest match, `*` / `$`, percent-encoding; an unreachable robots.txt or a 5xx skips the host as `robots_unreachable`, a 4xx allows it) and caps robots.txt / sitemap downloads; navigation stripping never drops `html` / `body` / `main` / `article` / `role="main"` and falls back to tag-only stripping when a class name would empty the page. Git `exclude_globs` are added to the built-in defaults (`default_excludes: false` turns those off; they match at any depth), URLs with embedded credentials and loopback ssh targets are refused, and imports stop promptly when the Job is cancelled. Repo source / config files are never read as training rows or used for task-shape detection. Rejected pages / files are reported by stable reason code; each document records where it came from (`source_url`, `path`, `commit`, licence).
- Added a **code chunker**: source and config files (Java, C#, Python, Go, JS/TS, C/C++, shell, Terraform, YAML, JSON, XML, SQL, INI, properties, Dockerfile) are cut at function / method / class boundaries with exact line ranges and symbols; code is cleaned verbatim (counted, not redacted) and passages are cited as `path:start-end (symbol)`. Source files can now also be uploaded.
- BM25 retrieval splits `camelCase` / `snake_case` identifiers into words (and keeps the whole identifier); all-caps plurals (`IDs`, `APIs`) and `5G` stay whole; indexes are stamped `BM25_TOKENIZER_VERSION = 3`, older ones are rebuilt automatically and `retrieve()` refuses a stale one instead of mis-scoring it.
- Dataset-import sources `web:` and `git:`; `DocumentResponse` now includes `license_info`, `source_url`, `path`, `commit`, `content_kind`, `language`; new setting `DOCUMENT_SOURCES_ALLOW_LOCAL_GIT` (default off).

### Latest Updates (March 5, 2026)

- Added **Model Registry** lifecycle with readiness snapshots, promotion gates, and deploy tracking.
- Added **Project Secrets** APIs with masked listing and encrypted-at-rest values.
- Added **Domain Profiles** with a typed contract and project-level assignment.
- Added **Domain Packs** (pluggable overlays + default profile pointer) with project-level assignment.
- Added **Domain Pack Hooks** for custom normalizers, validators, and evaluators.
- Added **Domain Profile Manager UI** (list/create/edit/assign) in project detail and profile selection during project creation.
- Added **Domain Pack Manager UI** (list/create/edit/assign) in project detail and pack selection during project creation.
- Added **Workflow Graph Preview** (read-only visual stage graph + step contracts) in project detail.
- Added **Workflow Contract Runtime (Phase 2)** APIs: graph validate, dry-run, run active step, and run history.
- Added **Visual Pipeline Editor (Phase 3)** with project-scoped graph contract save/load/reset and compile diagnostics.
- Added **Step Contract SDK v1** runtime fields (`runtime_requirements`) with compile/dry-run/run-step enforcement.
- Added **Typed Artifact Registry (Phase 4)** with versioned artifacts and pipeline run-step artifact publication.
- Added **Workflow Runner v1 (Phase 5)** with persisted DAG runs, node retries, dependency tracking, and backend adapters (`local`, `celery`, `external`).
- Added **Workflow Templates + Run Monitor UI (Phase 6)** on the Workflow page.
- Refactored project UI into a cleaner workspace IA:
  - `Pipeline` page
  - `Training Config` page
  - `Workflow Graph` page
  - `Domain Packs` page
  - `Domain Profiles` page
  - `Domain Contracts` legacy combined page (backward-compatible route)
  - `Pipeline Recipes` page
  - Training run monitoring remains in the `Pipeline -> Training` stage
  - left sidebar navigation with pipeline stages as submenu
- Added runtime transparency in split/training responses (applied profile + resolved defaults).
- Added **Duplicate-as-new-version** flow for domain profiles from the project UI.
- Added dedicated **Resolved Defaults** panels in Dataset Prep and Training tabs.
- Added queued **remote ingestion** jobs with task status/cancel APIs and WebSocket log streaming.
- Added queued **compression** jobs (quantize/merge/benchmark) with task status/cancel APIs and WebSocket logs.
- Added real GGUF/ONNX compression runtime paths in `backend/scripts/quantize.py`.
- Added **experiment comparison** API/UI for side-by-side metrics and loss-history visualization.
- Added auth UX updates for **SSO + local login** flow.
- Added training runtime generalization:
  - trainer backend abstraction (`auto`, `hf_trainer`, `trl_sft`)
  - task/data adapter contract (`causal_lm`, `seq2seq`, `classification`)
  - CUDA OOM auto-retry planner with progressive memory downscaling
- Added **Training Capability Matrix + Preflight**:
  - explicit config preflight endpoint before experiment creation/start
  - model/task/trainer compatibility checks
  - dependency/runtime/data-file preflight checks
  - start-time server guard that blocks incompatible training runs
- Added **Training Runtime Plugin SDK v1**:
  - runtime registry + catalog endpoint (`/training/runtimes`)
  - pluggable runtime selection via `training_runtime_id`
  - built-in runtimes: `builtin.simulate`, `builtin.external_celery`
  - legacy `TRAINING_BACKEND` remains default fallback for backward compatibility
- Added **Training Recipe System v1**:
  - built-in domain-agnostic recipe catalog (safe/balanced SFT, LoRA-fast, classification, seq2seq)
  - recipe resolve endpoint with runtime/profile default merge + optional preflight
  - Training UI recipe starter picker with one-click apply
- Added **Evaluation Packs + Auto Gates v1**:
  - built-in gate packs (`general`, `strict`, `fast-iteration`) + domain-profile-derived dynamic pack
  - project-level evaluation pack preference with fallback to defaults
  - experiment gate evaluation endpoint and pipeline status auto-gate summary
- Added **Pipeline Recipes / Blueprints v2 (Recipe Compiler)**:
  - built-in end-to-end recipes (`recipe.pipeline.sft_default`, `recipe.pipeline.lora_fast`, `recipe.pipeline.eval_gate`, `recipe.pipeline.program.sft_sweep`)
  - resolve/apply APIs that wire domain pack/profile, dataset adapter preset, training recipe, and evaluation pack
  - workflow resolution now supports `workflow.template_id` and executable `workflow.program` contracts
  - Recipe Compiler v2 supports conditional nodes/edges, failure-type retry policies, and loop/sweep node execution plans
  - task-profile-aware recipe recommendation context (`recommended_recipe_id`) for faster defaults on the Recipes page
  - persisted active recipe state + versioned reproducibility manifest artifact (`manifest.pipeline_recipe`)
  - dedicated `Pipeline Recipes` page in project workspace navigation (separate from Pipeline and Workflow pages)
- Added top-right **User Menu** in workspace:
  - profile identity (username/role)
  - logout action
  - runtime **Settings** modal for editable system config fields (persisted overrides)
- Added **Preflight Plan Suggestions** (`safe`, `balanced`, `max_quality`) with one-click config apply in Training UI.
- Added project-persisted training plan preference (`preferred_plan_profile`) so recommended profile choice is remembered per project.
- Added project/domain-pack adapter preset resolution for dataset split defaults (`adapter_id`, `adapter_config`, `field_mapping`).
- Added **Alignment Workbench baseline** for DPO/ORPO:
  - preference dataset import/filter APIs with apply-to-train support
  - judge quality keep-ratio guardrails in preflight
  - training-mode controls + alignment auto-filter settings in Training Config UI
  - native TRL pairwise objective runtime path (`DPOTrainer` / `ORPOTrainer`) in external training script
- Added **Universal Task/Dataset Adapter v3**:
  - typed adapter contracts (`task_profiles`, `preferred_training_tasks`, `output_contract`)
  - expanded built-in adapters: `rag-grounded`, `tool-call-json`, `structured-extraction` (plus existing chat/preference adapters)
  - richer preview diagnostics: `conformance_report`, `auto_fix_suggestions`, `inferred_task_profiles`, `raw_field_frequency`
  - `task_profile` wiring across adapter preview, project adapter preference, split manifests, workflow adapter-preview node, and remote import
- Added **Pipeline Recipe Executable Runs (Phase 18)**:
  - `POST /pipeline/recipes/run` to apply + execute recipe-backed workflow DAG (sync or async)
  - persisted recipe-run lineage records (`recipe_run_id -> workflow_run_id -> manifest snapshot`)
  - `GET /pipeline/recipes/runs` and `GET /pipeline/recipes/runs/{recipe_run_id}` for status/history
  - Recipe page now includes one-click run controls and recipe-run monitor
- Added **Recipe Run Control + Recovery (Phase 19)**:
  - `POST /pipeline/recipes/runs/{recipe_run_id}/cancel`
  - `POST /pipeline/recipes/runs/{recipe_run_id}/retry`
  - `POST /pipeline/recipes/runs/{recipe_run_id}/resume`
  - resume reuses completed nodes from source workflow run and can start from explicit node id
  - Recipes run inspector now exposes Cancel/Retry/Resume actions
- Added strict dataset contract checks in training preflight (task-shape coverage gate with actionable fix hints).
- Expanded Training experiment form controls for memory/runtime tuning (`gradient_accumulation_steps`, `max_seq_length`, `save_steps`, `eval_steps`, `fp16`/`bf16`, `flash_attention`, `sequence_packing`).
- Added split backend dependency profiles for CPU/GPU installs (`requirements-base.txt`, `requirements.txt`, `requirements-gpu.txt`, `requirements-gpu-cu128.txt`).
- Added live external-training telemetry streaming (`epoch`, `step`, loss metrics) from Celery worker to Training dashboard via WebSocket.
- Added **Deployment Target SDK v1**:
  - deployment target catalog for exporters/runners (`HF`, `GGUF`, `ONNX`, `TensorRT`, `vLLM`, `TGI`, `Ollama`)
  - export-time deployability validation (artifact profile checks)
  - optional local runner smoke tests and explicit deployment reports in export manifests
- Added **Conformance + Scale Test Matrix (Phase 20)**:
  - automated backend matrix for `adapters x runtimes x model families x task types`
  - scale checks for adapter mapping quality at larger sample sizes
  - explicit guardrail assertions for model-task incompatibility and dataset-contract incompatibility
- Added Alembic revisions:
  - `20260305_0002` for registry + secrets tables
  - `20260305_0003` for domain profiles + project binding
  - `20260305_0004` for domain packs + project binding
  - `20260305_0005` for artifact registry table
  - `20260306_0007` for project training preference persistence
  - `20260306_0008` for project dataset adapter preset persistence
  - `20260307_0009` for project evaluation pack preference persistence

---
