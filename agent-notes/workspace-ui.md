# Workspace UI, Coach, visual pass, demo videos

Detail notes split out of CLAUDE.md. Read when working on this area.

- **Coach Mode** — `coach_service.py` emits stage suggestions
  (`data` / `cleaning` / `gold_set` / `synthetic` / `dataprep` / `training` /
  `eval` / `export` — the last three stages added in Wave 3b: synthetic mix +
  pending review, split missing/stale/test-too-small/leakage, export-vs-base
  verdict from `eval_summary_service`). Shared nudges are stage-prefixed
  helpers (`_split_leakage_nudge`, `_synth_review_pending_nudge`). Navigate
  targets `pipeline-tab` (`params.tab`) and `dataprep-split` (+ the
  `brewslm:open-split-form` event for same-page clicks). Frontend
  `CoachStrip` polls + renders via `CoachSuggestionCard`. Actions:
  `run_playbook`, `navigate`, `augment_from_cluster`.
  `NAVIGATE_TARGET_URLS` in `CoachSuggestion.tsx` maps target names
  to URLs.
- **One intake + one Coach (Wave 3)** — `beginner_mode` defaults **True**
  (model column, `ProjectCreate`, manifest `WorkflowSection`). Creation =
  "+ New Project" (brief-driven) or a sample (DemoProjectTiles /
  ProjectTemplateGallery under "Or start from a sample project"); manifest
  import is a link in the create dialog's Advanced section. Magic Create's UI
  and FirstRunCheatSheet are gone (the `/projects/magic-create` API remains).
  Every path navigates to `/project/{id}` (the Overview).
  **Workspace IA (Epic U phase U1, 2026-10-17)** — `utils/workspaceRoutes.ts`
  is the one map: six destinations `Overview` (`/project/{id}`,
  `ProjectOverviewPage`: stage, next step, latest run via
  `/evaluation/summary`, shipped export + Hub link via `/export/list`, ≤ 3
  Coach nudges, collapsed "Full plan" = checklist + QuickstartCard), `Data` /
  `Train` / `Results` / `Ship` (`ProjectStagePage destination=…`: the old
  pipeline tabs as steps, `?step=<tab>`, `STEPS_BY_DESTINATION`, header
  "related" links to the tool pages not folded in yet) and `Settings`
  (`ProjectSettingsPage`: inline-tips switch, beginner flag = "start advanced
  sections collapsed", manifest export, links to Automation / Domain /
  Studios). Sidebar = `ProjectSidebar` (six `NavLink`s, `data-testid="nav-<dest>"`,
  ⌘K hint, collapse); nothing is hidden by beginner mode any more. Retired:
  `/guide`, `/wizard` (the guided Autopilot page; Planner stays at
  `/autopilot`), `/pipeline/:tab`, ProjectPipelinePage, ProgressChip,
  CoachToggle, all gamification (frontend `components/gamification`, backend
  `api/gamification.py` + service; the `projects.gamification` column stays,
  unused). `LegacyRouteRedirect` + `legacyRedirectPath` send old URLs to the
  new ones with search + hash kept — the bell, `CoachSuggestion`
  `NAVIGATE_TARGET_URLS`, docs and demo specs still emit `/pipeline/<tab>`
  until U6 rewrites them; new code uses `stagePath(projectId, tab)` /
  `projectPath(projectId, dest)` (flowGuide already does).
  **Train = the form (Epic U phase U2, 2026-10-17)** — `pages/ProjectTrainPage.tsx`
  mounts `TrainingPanel layout="form"`: `components/training/TrainForm.tsx`
  (presentational; page one = base model + Browse, task shape read-only +
  Change, seeds radio 1 / 3, run name, readiness line, one Train button;
  "More options" = collapsible sections rendered only while open, `?open=<id>`
  / `openNonce` opens one) + the runs list below. The panel's form branch (before
  the classic `return`) builds its own sections (`model`, `lora`, `seeds`,
  `checks`, `cloud`, `presets`) from the same state as the classic setup tabs
  and appends `moreOptions` from the page (`plan` = the former
  /training-config panels, `models` = ProjectModelsPage, `task` =
  ProjectRecipePickerPage, `pipeline-presets` = ProjectRecipesPage,
  `adapters` = ProjectAdapterStudioPage, `tokenization` — those four pages
  take `embedded` and render inside the form; `afterForm` holds the
  documents→Q&A card). Readiness = `runPreflightPreview` on mount and 600 ms
  after a base-model change; Train = `handleCreate(name || run-<date>)` →
  `PreRunConfirmModal` → `handleStart`. Prefill: after the runs list loads,
  the latest non-baseline run's config via `applySuggestedConfig` + its
  `base_model` column, else `project.base_model_name`. Retired routes
  `/training-config` (+ `#base-model`, `?expand_multi_seed=1`, the gaps /
  archetype anchors), `/models`, `/recipe-picker`, `/recipes`,
  `/adapter-studio`, `/pipeline/tokenization` redirect into
  `trainSectionPath(projectId, id)` (`workspaceRoutes.TRAIN_OPEN_BY_LEGACY_PATH`);
  Train has no step bar (`STEPS_BY_DESTINATION.train = ['training']`). Tests:
  `TrainForm.test.tsx`, `TrainingPanel.form.test.tsx`, `ProjectTrainPage.test.tsx`.
  **Results = four workflows (Epic U phase U3, 2026-10-17)** —
  `pages/ProjectResultsPage.tsx`: tabs Verdict / Failures / Compare / Try it
  (`?tab=`, `workspaceRoutes.RESULTS_TABS`; `resolveResultsTab(search, hash)`
  maps the old eval anchors via `RESULTS_TAB_BY_HASH`), one
  `<EvalPanel workflow={tab}>` kept mounted across tabs (the selected run and
  its results persist). **One selected run** (2026-10-19): the panel opens on
  `runSelection.defaultRunId` (newest COMPLETED run that is neither a
  `config.is_baseline` lift-check baseline nor a seed child — the summary's
  rule) on every tab; the summary card's `onResolved` re-syncs it until the
  user picks a run (`userPickedRunRef`). The picker shows `#id · name` + a
  "baseline" badge; the Verdict pass/fail headline (`eval-verdict-gate`) and
  `ScorecardPanel runLabel` name the run (`runLabel`). The page-level
  `pipelineStatus.auto_gate` card is gone (it named the newest experiment —
  the baseline); `/pipeline/status` `auto_gate` now skips baselines / seed
  children too. `EvalPanel` gates its blocks with `wf('<workflow>')`
  (`workflow` null = the old one-page layout the older tests still mount, with
  the Advanced toggle and the runs/workbench sub-tabs). **Layout since UI simplification phase 1 (2026-11-20)**:
  Verdict = `VerdictCard` (top), summary card (`hideTryBox`), run picker, "+ Score this run"
  (`EvalRunForm`, phase 2), `AdvancedSection` "Test sets" (`verdict-test-sets`: `EvalSetsCard`), aggregate badge, `AdvancedSection` "Scores" (`verdict-scores`, kept
  mounted: Metric History + every per-result panel + PerSliceMetricsTable),
  `HowItsScoredCard`, `AdvancedSection` "Details" (`verdict-details`: the
  `eval-verdict-gate` rule report, SftLiftPanel, ScorecardPanel, runtime card,
  safety scorecard); Failures = clusters, schema banner, failure rows, the judge
  side-by-side, EvalGaps, then `AdvancedSection` `failures-tools` (kept mounted:
  ProbePack, the rule-set card with slices, the remediation planner, behavioral
  table, active learning, reroute, drift queue, skill radar, AnswerKeyReview and
  GoldSetWorkbench `<details>`). Details in the phase 1 section below;
  Compare = ExternalBenchmarksPanel, Frontier, AutoRag, StudentTeacher (+ the
  page mounts `ProjectEvalComparePage embedded` when `?a&b`); Try it = the
  page mounts `ChatPlaygroundPanel` and the panel renders `RedactionTryBox`
  for a span run. Retired with redirects: `/eval/compare` →
  `resultsPath(id, 'compare', search)`, `/playground` → `try`,
  `/pipeline/eval#anchor` → its tab. ProjectPlaygroundPage deleted; the
  compare page kept as an embeddable component. Tests:
  `EvalPanel.workflows.test.tsx`, `ProjectResultsPage.test.tsx`.
  **Data = five steps (Epic U phase U4, 2026-10-17)** — `pages/ProjectDataPage.tsx`:
  Sources / Prepare / Labels / Generate / Split (`?step=<tab>`), each = the
  old panel in `layout="stepper"` (short form on top + a
  `components/shared/DetailsDrawer` of lazy sections, `?open=<id>`) plus the
  page's `extraDetails` (the former Data Studio panels, mapped per step; the
  annotation job list = `AnnotateListView embedded` under Labels). The five
  panels keep `layout="classic"` (unchanged tree) and in stepper mode gate
  with `stepper`; their secondary blocks became consts (`docsSection`,
  `entriesCard`, `overviewView` / `adaptersView` / `splitView`) reused by
  both layouts. Labels adds `TaskShapeCard` (own fetch of `/task-shape`) and,
  for span projects (`isSpanProject`: recipe `scoring_mode: span_set` OR
  runtime_config span definitions / custom patterns / compliance bundles —
  project 31's snapshot still says qa-sft), `LabellingConventionsCard`
  (bundles + definitions + add-type). Retired with redirects: `/data-studio`
  (+ `#section` → `DATA_OPEN_BY_STUDIO_SECTION`), `/annotate` (the labeler
  `/annotate/:job_id` stays), `/pipeline/<tab>#synth-review-queue|#label-noise-review`
  → their Details section (`DATA_OPEN_BY_HASH`). ProjectDataStudioPage and
  the coach rail panel are gone; `dataStudioTargetPath` maps the studio
  panels' `onOpenTarget` targets. Tests: `DataPanels.stepper.test.tsx`,
  `ProjectDataPage.test.tsx`, `DetailsDrawer.test.tsx`.
  **Ship = one flow (Epic U phase U5, 2026-10-17)** — `pages/ProjectShipPage.tsx`:
  stages Package / Publish / Deploy / Monitor (`?stage=`, `workspaceRoutes.SHIP_STAGES`,
  `shipPath`), one `<ExportPanel stage>` kept mounted (plans and the releases
  list persist). `ExportPanel` gates with `st('<stage>')` (`stage` null = the
  old one-page layout the older tests mount): Package = the export form ("New
  export"), a **Releases** header card + the exports table ("Exports") + the
  registered models table; Publish = a card with one Hub-publish row per
  completed HF export (same `handlePublishHub` / `hubRepoIds`; the row form
  stays in the classic layout only); Deploy = the serve / deploy plan cards
  (also shown on Package since they are requested there). The page adds
  `SeedPickCard` (the summary's `seed_group_leader` → `SeedShipTable`) and an
  Options `DetailsDrawer` (`compress` = CompressionPanel, `manifest` =
  ProjectManifestPage embedded) on Package, `ProjectDeploymentsPage embedded`
  on Deploy, `ProjectObservabilityPage embedded` on Monitor. Retired with
  redirects: `/pipeline/compression` → `?open=compress`, `/pipeline/export`,
  `/deployments` → deploy, `/observability` → monitor (hash kept), `/manifest`
  → package + manifest (`/manifest/import` for new projects stays).
  `STEPS_BY_DESTINATION.ship = ['export']`; ProjectStagePage is deleted (its
  CSS lives on as `pages/stagePage.css`, shared by Data and Ship). Tests:
  `ExportPanel.stages.test.tsx`, `ProjectShipPage.test.tsx`.
  **Gateway cards (G11)** — Deploy also mounts `components/gateway/GatewayCard`
  ("Use from Claude Code": model ids with project + run + base, `.mcp.json` /
  curl / OpenAI snippets with `${BREWSLM_API_KEY}`, API keys shown once, gateway
  settings, named-tools editor with the `@recipe` template checkbox, recent
  requests) and `MultiLoraCard` (vLLM multi-LoRA plan, start, route the gateway).
  Client `api/gateway.ts`; 400s read `detail` + `errors ?? metadata.errors`.
  Tests: `GatewayCard.test.tsx`.
  **Guidance is inline (Epic U phase U6, 2026-10-17)** — the Coach bar,
  `CoachStrip`, `CoachToggle` and `coachLocationFor` are gone. The backend
  `coach_service` stages/rules are unchanged; the frontend renders them as
  `components/coach/CoachNotices` (one per destination: Data per step's
  stage, Train `training`, Results `eval`, Ship `export`; `CoachSuggestionCard`
  with its action + "why?" trace, an × that remembers the `rule_id` in
  localStorage `brewslm.coach.dismissed.<pid>.<stage>`, "Show N more") and
  `StepHelp` (the step's `coachContext.TAB_TIP` + `TabVideoLink`, "Got it"
  key `brewslm.coach.tip.<pid>.<tab>`; Results and Ship pass `sub=<tab|stage>` → `coachContext.SUB_STEP_TIP` + `tabVideos.SUB_STEP_CHAPTER`, key `<tab>.<sub>`). The Overview lists ≤ 3 nudges.
  `useCoachMode(projectId)` (on by default, override in `coachModeStore`) is the
  one switch — Settings → "Show inline tips". Beginner mode hides nothing
  anywhere (the `flowGuide` domain-first detour is gone too).
  **Only one layout per panel.** `TrainingPanel` = the form (props
  `hideCreateControls` / `forceCreateVisible` / `hideExperimentList` /
  `setupMode` / `layout` deleted with the setup tabs, journey strip and
  workspace tabs); `EvalPanel` requires `workflow`; `ExportPanel` requires
  `stage` (the in-row Hub form is gone — Publish stage only); the five data
  panels render form + `DetailsDrawer` (no `layout` prop; `DatasetPrepPanel`
  lost its view tabs and the TaskShapeConfirmCard, which lives on Data →
  Labels). **Settings = the long tail as sections** (`SETTINGS_SECTIONS`,
  `settingsPath`, `?open=`): Autopilot Planner, Workflow Builder, Domain Packs,
  Domain Profiles, Extension Studio render embedded (`embedded` prop), plus
  `decisions` = `components/autopilot/DecisionLogPanel` (the Autopilot
  decision log; it was a floating drawer on every page until 2026-10-18).
  Blank project in the Autopilot Planner: `evaluate_newbie_autopilot_dataset_readiness(has_any_data=False)`
  (no dataset with rows, no documents: `api/training._project_has_any_data`) gives `NO_DATA_BLOCKER`
  ("No training data yet … Add data first") + an "Add data" fix to `/project/{id}/data`; the one-click
  run skips auto-prepare (`repairs.dataset_auto_prepare.no_data`). The page shows an **Add data** link on that
  blocker and an "add data first" line on a not-started run.
  The
  routes `/autopilot`, `/workflow`, `/domain`, `/domain/packs`,
  `/domain/profiles`, `/extensions` redirect (`SETTINGS_SECTION_BY_LEGACY_PATH`);
  the Domain Packs section also carries pack *content* (`api/domainPacks.ts`):
  `DomainPackImportPanel` (zip -> preview with issues grouped by code, summary,
  diff / unchanged -> import) and `DomainPackContentList` (packs with content,
  "Create project from this pack" via `window.location.assign`, "Export this
  project as a pack"); `ProbePackPanel` also lists *your checks* (source label,
  severity, critical tag, `by_source`, critical failures) with a paste-YAML
  `UserProbeEditor` (validate / save / delete); the `probe_critical_failures`
  gate sits under Scorecard's "Independent ruler";
  ProjectDomainPage deleted. **The ⌘K palette** renders
  `utils/workspaceCommands.workspaceCommands(projectId)` — every destination,
  Data step/section, Train section, Results workflow, Ship stage, Settings
  section, generated from `workspaceRoutes`; add a surface there, never in
  the palette. In-app links use the `*Path` helpers; `LegacyRouteRedirect`
  stays for bookmarks and docs.
  **The visual pass (Epic U phase U7, 2026-10-18)** — `frontend/src/index.css`
  is the whole palette: a gray ramp, ONE accent hue (`--accent` blue:
  primary buttons, active nav/tab, links, focus rings, selected states;
  `--accent-text` / `--accent-bg` / `--accent-border` / `--accent-contrast`),
  three status colours (`--color-success|warning|error` + `-bg` / `-border`),
  a type scale (`--font-size-2xs…3xl` + `--leading-*`) and the spacing scale.
  **Dark theme**: `html[data-theme="dark"]` mirrors the ramp and re-shades
  the accent/status tokens; `stores/themeStore` (`light` default, `dark`,
  `system`; localStorage `brewslm:theme`; `initTheme()` in `main.tsx` before
  first paint) writes the resolved value; the switch is Settings → Appearance
  and the ⌘K "Switch to the dark/light theme" command. Because `--gray-0`
  flips, text on a coloured surface uses `--accent-contrast`, never
  `white`. **No component writes a colour literal, an inline `style={{…}}`
  or an emoji**: `styles/visualPass.test.ts` lists the clean files and
  asserts no stylesheet but `index.css` has a hex — add new files to the
  list, never remove one. Layout one-offs use `styles/utilities.css`
  (`u-stack`, `u-row`, `u-grid-2`, `u-m0`, `u-meta`, `u-hint`,
  `u-card-title`, `u-icon`…; loaded AFTER index.css from `main.tsx` so a
  utility wins a specificity tie); data-driven widths are bucketed classes
  (10 % steps); icons are `lucide-react`. **Empty + loading**:
  `components/shared/EmptyState` (lucide icon node, `size="compact"` inside
  a card, testid `empty-state`) and `LoadingState` (skeleton lines +
  sr-only label, testid `loading-state`; `Skeleton variant=…`) are the only
  two ways to say "nothing yet" / "fetching"; a one-line "none" inside a
  dense table stays text (`u-meta`). `StepFooter.nextStepIcon` is unused.
  **Charts** (2026-10-18 follow-up): series colours are `--chart-1…6`
  (1 = the accent; the only place a second hue is allowed), axes / grid /
  tooltip `--chart-axis|grid|tooltip-bg|tooltip-border`, inline highlight
  tones `--highlight-1…9` (the span labeler's entity types,
  `.span-tone-N`); recharts takes them as `stroke="var(--chart-grid)"` and
  `<Tooltip wrapperClassName="chart-tooltip">` (rule in index.css). The
  lock-in test also refuses any colour literal (hex / rgb / hsl) in a clean
  file. Template icons are lucide NAMES in each
  `backend/data/project_templates/*/manifest.json` (`"icon": "mail"`,
  default `package`; `ProjectTemplateGallery.TEMPLATE_ICONS` maps them,
  unknown → `Package`). **Disclosure glyphs**: no file under `src` draws a
  triangle (U+25B2–U+25C5: ▸ ▾ ▶ ▼ ▲ …) — chevrons are lucide
  `ChevronRight` / `ChevronDown` (`className="u-icon" aria-hidden="true"`;
  a `<details>` summary rotates one `ChevronRight` on `[open]`), trend arrows
  `ArrowUp` / `ArrowDown`; the lock-in test scans every .ts/.tsx/.css. Gate
  status text (ScorecardPanel) = `StatusText` (tone class + lucide icon), no
  emoji. Demo videos
  02 (quickstart) and 03 (pipeline walkthrough — the in-app "Watch the
  walkthrough" catalogue `data/tabVideos.ts`) were re-recorded on the new
  workspace via `tests/demo-recordings/0{2,3}-*-narrated.spec.ts` +
  `tts/narration_tts.py` (Orpheus through Ollama, no FastAPI bridge);
  outputs under `docs-demo/recordings/` (gitignored), upload is manual.
  Every narrated spec (02–06, 09–12, 14) is on the workspace and every
  generator uses `narration_tts.generate_video`. **Record only against a
  scratch stack**: `docs-demo/scripts/ffmpeg/scratch-stack.sh up` (backend
  :8031 + vite :5176 on a fresh `/tmp/brewslm-demo-stack` DB, auth on,
  simulated training; `up --real` adds its own Celery worker on Redis DBs 4/5
  for real `train.py` — videos 10–12 need real weights) → `record-narrated.sh NN`
  (playwright on `DEMO_BASE_URL`, default :5176; `playwright.config.ts` reads
  it) → concat → trim the pre-roll each spec writes to `tts/audio/vNN-preroll.json`
  → `docs-demo/recordings/raw/<slug>-narrated.mp4`. `DEMO_STRICT_BEATS=1`
  makes a check run fail on a missing beat element (a recording only warns).
  Specs call the API by relative `/api/…` (never `localhost:8000`). 02–06,
  09–12 and 14 re-recorded 2026-10-18 (awaiting upload; paste-ready sheet
  `docs-demo/youtube/upload-2026-10-18.md`, thumbnails from
  `docs-demo/youtube/thumbnails-source.html` via `youtube-thumbnails.spec.ts`).
  Narration that quotes a number (steps, seconds, F1, MB, what the model
  replied) must be re-read against the recorded run — the 09–12 scripts had
  carried an older take's figures. 09–12 run in order on ONE `up --real`
  stack (10 scores 09's run, 11 compresses it, 12 serves 11's GGUF and
  re-registers the host Ollama alias `slm-supportfaq`; 11 asserts the export
  `completed`, 12 reads the GGUF from `compressed/exp-<run>/`, sets the playground
  temperature to 0 (`data-testid="playground-temperature"`) and runs
  `ollama stop` before the prompt, then logs `[v12] reply:` — greedy decoding
  is repeatable only per Ollama STATE: a cold load and a warm model with the
  prompt cached give two different (each repeatable) replies, so the spec
  always asks a cold model; the narration quotes that reply). Videos 08–10
  were recorded BEFORE training was pinned deterministic (F1 0.2405); under
  the pin spec 09's run is identical on every stack (weights_sha256
  `88dfc789…`, spec 10 F1 **0.2419**, 2 fresh stacks) — a re-record must
  re-read video 08's F1, video 09's GGUF size and video 10's reply. Spec 14 is
  slides over `file://` and needs no stack (`DEMO_BASE_URL=http://localhost:5173`
  just satisfies record-narrated's up-check). The auto lift check
  adds a baseline run above the trained one, so specs target a run by
  `data-testid="training-run-<id>"`, never the first row. API reference pages
  (`slm-docs/docs/reference/api/`) are regenerated per `slm-docs/scripts/README.md`
  (dump with DATABASE_URL/DATA_DIR on a scratch DB, then `npx docusaurus build`).

## Judge picker, subscription backend and hard API budget (G9)

- Settings has a "Judge model" card (`components/settings/JudgeModelCard.tsx`, `api/evalJudge.ts`): select from
  `GET /eval-judge` options, enabled toggle, the resolved judge with a billing badge (local / metered /
  subscription) and an `EVAL_JUDGE_BACKEND` override notice. PUT returns the same shape.
- `GenerationModelPicker` adds a "Local CLI (subscription) — <detail>" option when `/synthetic/backends` reports an
  available `billing === "subscription"` backend; the pin is that backend's `describe`. `PlaybookPickerPanel` shows
  a "subscription — not billed per call" chip + the backend `detail` for the pin that will run.
- `LlmSpendCard`: subscription rows show "subscription" instead of $; a separate "Subscription CLI" line; the budget
  copy is a hard stop for cloud API calls; "All projects this month" appears when `global.monthly` is set
  (read from `/llm-budget`).
- `NotificationBell`: a running Job whose message starts "waiting for usage window" gets a "paused" chip.
- A 402 `llm_budget_exceeded` start surfaces through the existing `extractApiErrorMessage` toasts (envelope message).

## Label taxonomy card (G3)

`components/data/LabelTaxonomyCard.tsx` (client `api/labelTaxonomy.ts`) sits on the Data page's labels step for every project, collapsed by default. Header shows the taxonomy version chip + entry count; body: searchable parent/child tree (id, title, aliases, crosswalk `namespace: refs`, attrs), "Try a label" resolver (`/resolve`, shows id + matched_by), editor (file upload or paste, `crosswalk_from` comma list, Validate / Save with 422 `metadata.errors` grouped by code), history with Restore, Delete with confirm. `DatasetImportWizard` shows "Labels resolved with taxonomy <version> (<n> entries)" / "No taxonomy: labels kept as-is" for the findings mappers (`code_label_to_findings`, `fix_pair`, `contrast_functions`) from the preview result's `taxonomy`.

## Code-findings surfaces (phase G4)

- Ship > Package: `GuidedJsonCard` (collapsible "Structured output hint", `GET /projects/{id}/guided-json`; renders nothing when null).
- Data > Synthetic playbooks: recipe `code-findings` adds the `seeded_defect` mode; `CodeFindingsParamsForm` sends `params` (label from the taxonomy, or a typed free label when the project has none; attrs prefilled from the entry; optional dataset picker from `GET /projects/{id}/datasets` (answer-key datasets filtered out) + a column picker from `.../datasets/{id}/columns` + sample size, each shown with its id). `SynthRejectSummary` shows the finished job's `reject_summary` / `rejected_examples`; the bell appends "N rejected".
- Eval: `FindingSetMetricsPanel` (metrics, per-label table, no confusion matrix) and `FindingSetRowView` (matched / missed / spurious lines with path, language, start line) for `scoring_mode === 'finding_set'`. Probe pack labels `returns_empty_findings` / `findings_stable_vs_base` (+ `base_had_no_findings` note); code probes carry a `{code, language, path, start_line}` input.

## Code-remediation surfaces (phase G5)

- Train > "Remediation checks" section (`pages/ProjectTrainPage.tsx`, only when the project recipe is `code-remediation`; `trainPageUtils` id `remediation`): `components/training/RemediationConfigCard.tsx` (clients `api/remediation.ts` `getRemediationConfig` / `putRemediationConfig`, `api/codeChecks.ts` `listCodeChecks`). Detector picker labelled `<project> · run <id>` (plus allow-listed command detectors), code-check list with ready / not installed / not allowed and the network-isolation mode, judge toggle, retrieve-passages toggle + top_k (1..10). Server validation errors show under the Save button.
- Data > Synthetic playbooks: recipe `code-remediation` adds `remediation_from_findings` (`CodeRemediationParamsForm`: optional findings dataset, answer-key datasets filtered out, shown with its id; `top_k`) and `code_examples` (`per_chunk` 1..5; training-only rows). `PARAM_MODES` in `PlaybookPickerPanel` lists the modes that send `params`.
- Eval: `RemediationMetricsPanel` (pass / applies / parses / resolved / citations rates, "not evaluated" states when no detector, no checkable rows or no passages, patch-failure reasons, judge score, detector error) and `RemediationRowView` (finding, predicted patch as a diff with added/removed classes, status chips, check messages, passages with the cited ones marked) for `scoring_mode === 'remediation'` or rows carrying `row_patch_applies`. `liftEvidence` names the `remediation_pass_rate` headline for `EvalSummaryCard` / bell / lift panel.

## Structured-plan surfaces (phase G6)

- Train > "Plan schema and rubric" section (only when the recipe is `structured-plan`; `trainPageUtils` id `structured-plan`): `components/training/StructuredPlanConfigCard.tsx` (client `api/structuredPlan.ts`). Multi-file JSON-Schema upload (read client-side, file name = key) with a root select; saved root + file count + short sha, validation errors and whether the guided-decoding hint is set; rubric upload (.yaml/.yml/.json) with its id + short sha and a criteria table (weight, critical); a JSON-pointer editor (one per line); judge and retrieve-passages toggles + top_k (`retrieve_passages` is only sent once the user toggles it, so the auto default stays); a paste-a-JSON box validated against the saved schema. 422 `metadata.errors` list under Save.
- Data > Synthetic playbooks: recipe `structured-plan` adds `scenario_grid` (`ScenarioGridParamsForm`: matrix file (markdown table or CSV) read client-side, preview of dimensions × values and the cell count with how many will be sampled; `max_cells` 1..200, `scenarios_per_cell` 1..5, `seed`, `extra_instructions`). The preview parse mirrors the backend rule (each column a dimension); the server stays authoritative. Run is disabled until the matrix parses.
- Eval: `StructuredPlanMetricsPanel` (after the remediation panel, for `scoring_mode === 'structured_plan'`, excluded from the classification chart): pass / schema-valid / valid-JSON rates, top schema errors grouped by rule + pointer, per-pointer match / support / missing (answer key has it) rates, a per-criterion rubric table (weight, critical badge, mean, scores 1-2, rows) with judged/unjudged counts, a provenance line (schema root + short sha, rubric id + short sha, judge label), "no judge" / "not run" / "not evaluated" states, and a warning when the schema changed since the split was prepared. `StructuredPlanRowView`: status chips, schema errors, predicted vs expected per pointer ("any of" for acceptable sets), rubric scores with critical failures highlighted. `liftEvidence` names `structured_plan_pass_rate` "plan pass rate".

## Tool catalog and tool-call scoring surfaces (phase G7)

- Data > Labels step (`step=goldset`): a collapsed `components/data/ToolCatalogCard.tsx` (`#tool-catalog`, every project, after `LabelTaxonomyCard`; clients `api/toolCatalog.ts`).
  - Header: tool count, allowed count and catalog version.
  - Provenance line: version, source kind + label, command/URL/filename, server name + version, imported_at and flags version.
  - Import tabs: local MCP server (command one argument per line + env rows), HTTP MCP server (URL + header rows) and OpenAPI file. Secret values are password inputs that the API never returns. Stored names are prefilled with blank values; blank keeps the stored secret (sent as `null`).
  - Table: each tool's signature, a read-only / mutating / destructive badge, an allowed checkbox and a read-only override select (disabled for destructive tools; `null` = use the server hint).
  - Also bulk allow / deny / deny-mutating, history with restore, and clear.
  - Server errors (`MCP_STDIO_DISABLED`, `INVALID_TOOL_CATALOG`, `MCP_SERVER_ERROR`) show their code and message.
- Results › Verdict › How it's scored › Change how it's scored (was Train > More options until UI phase 4; `/train?open=tool-scoring` redirects) "Tool-call scoring" (`components/training/ToolScoringConfigCard.tsx`, `api/toolScoring.ts`, `ScoringSettingsSection`), for `structured-plan` recipes and projects whose backend-reported `effective_task_profile` (from `GET .../tool-scoring-config`: `rcs.effective_task_profile` = prepared manifest, else latest run, else saved dataset preset, else recipe; `selected_recipe.task_profile` only until it loads) is `tool_calling`. Never read the task profile from project-response fields alone: until 2026-11-19 the project response had no `dataset_adapter_preset`, so the section never appeared (P5 gap 4). The response now carries `dataset_adapter_preset` (the saved preset, e.g. a demo's `suggested_brief` for Autopilot), but it is not the effective profile.
  - Checks, steps pointer and field names (plans only), arg keys.
  - A dry-run toggle labelled "Read-only tools only; never calls mutating tools"; a 422 shows when dry-run is disabled server-side.
- How it's scored › Change how it's scored "SQL execution match" (`components/training/SqlExecutionConfigCard.tsx`, `api/sqlExecution.ts`; `/train?open=sql-execution` redirects), shown for every project (opt-in; see `agent-notes/evaluation.md`).
- Eval: `ToolCallMetricsSection` renders inside `StructuredPlanMetricsPanel` whenever `metrics.tools` exists. `ToolCallMetricsPanel` is the standalone card for `scoring_mode: tool_call` and is excluded from the classification chart.
  - Both show the rates ("not applicable" for null), a per-tool table, unknown tools, top errors, the dry-run summary and the catalog provenance line.
  - `ToolCallRowView` shows each call with known / allowed / args-valid / args-match / dry-run badges (or the skip reason) and arg errors.
  - `liftEvidence` labels `tool_call_pass_rate` "tool-call pass rate" and `tool_call_valid_rate` "valid tool-call rate".
- NotificationBell: `tool_catalog_import` jobs deep-link to the catalog card. The outcome is "N tools imported from <kind> (catalog version X)".

## Multi-turn chat surfaces (G8)

- `MultiTurnMaskingCard` (training run view, `TrainingRunView`): assistant token share, assistant turns, truncated /
  dropped, mask fallbacks; "approximate" tag for simulate. Reads the first of `config._runtime.multi_turn`,
  `training_report.multi_turn`, `runtime_environment.multi_turn`; hidden when none.
- `EvalPanel` run form: "Judge every assistant turn (multi-turn rows)" toggle sends `dialogue_judge: true` (omitted
  when off). `DialogueJudgePanel` renders `metrics.dialogue_judge` (mean, worst-turn mean, by-turn table).
- `PreferencePairsFromEvalCard` (shown for a result with judged rows: `metrics.judge.judged > 0` or `metrics.dialogue_judge.rows > 0`,
  `hasJudgedRows`; labelled Run # / dataset / result id) posts `/alignment/preference-pairs/from-eval {eval_result_id, include,
  allow_test_split?}`; the "Allow test split (measures memorisation)" checkbox shows only when the eval's split (`dataset_name`) is
  `test` / `gold_test`, and the reply's `warning` is shown. Shows created + name, `dataset_path` (`dataset_id` is null) with "To train
  DPO/ORPO on these pairs, set it as the alignment dataset path in Train.", the skip breakdown by reason, and the 400 breakdown (read
  from `skipped`, `metadata.skipped` or `detail.skipped`).
- `DialogueJudgePanel`: `{skipped: reason}` renders "Per-turn judge skipped: <reason>"; `replay_cap_hit` shows how many rows were not replayed.
- `MultiTurnParamsForm` for synth mode `documents_multi_turn` (turns_min/max (2-8), passages_per_dialogue,
  dialogues_per_passage_group); mode added to `SynthMode`, `PARAM_MODES`, `MODE_LABELS`.

## Pipelines page (G10)

Top-level, not under a project: `/pipelines` (list + "New pipeline"), `/pipelines/new`, `/pipelines/:id[?run=<run_id>]`
(`pages/PipelinesPage.tsx`, `pages/PipelineDetailPage.tsx`, parts in `components/pipelines/`: `PipelineEditor`,
`HeldoutCard`, `RunsCard`, `PipelineResults`, `PipelineMembership`; client `api/pipelines.ts`). The top bar carries a
plain `<a href="/pipelines">` (no Router needed, so TopBar stays mountable in isolation).
- Editor save surfaces the 400 envelope's per-field errors (`parseGatewayError`); a field `stages[i]....` renders inside
  stage i, others at the top. Model select = project default / base / `:run-N` from `gateway/settings.servable_runs`.
- Run = `POST .../runs?async_job=true` (Job kind `pipeline_eval`, project-less). `NotificationBell.jobDeepLink` reads
  `params.pipeline_id` + `params.run_id` (falls back to `result`) and opens `/pipelines/<id>?run=<run_id>`; the outcome
  summary reads `result.rows`, `result.share_fully_correct`, `result.attribution.first_wrong_stage`.
- Results: stage cards (isolated view only for stage 2+; unscored-by-reason, lost-to-error / key counts, isolated
  `skipped_rows`; a `StageScoring` list for judge ran / not run, `scoring_error`, item comparison), headline (share of
  SCORED final items + an "N left out" note for `unscored_items`), attribution `BarChart` (colours `var(--chart-N)`,
  plus a text list for the same data), rows table, truncated message (names the row), `gold_mismatch` notice.
- A run file without results (`status` running / failed / cancelled / stale, or no `stages`) renders a status panel
  with its `error` (`isFinishedRun` guards `PipelineResults`). RunsCard lists the status and reloads after a start;
  the list page shows the last run's status and a "project missing" badge for a deleted member project.
- Editor: max tokens is sent as typed (empty = server default 512; 0 is not coerced, shows an inline 1..4096 hint and
  the server's field error); a hint under Output says to use text for bare-label outputs. A member project's Results page shows
  "Part of pipelines: …" from `GET /projects/{id}/pipelines` (hidden when empty).
- Tests: `pages/Pipelines.test.tsx` (recharts mocked to expose the chart data), bell cases in `NotificationBell.test.tsx`.

## Row context chip (hardening Phase 2, 2026-11-19)

- `components/shared/RowContextChip.tsx` (types `types/rowContext.ts`): renders nothing without `row_context`;
  chips "Own system message" / "Own tool list (N)" / "Project catalog" / "not used by this task: …", expandable to
  the system text and tool signatures ("not allowed by project"). Accepts the eval view and a raw row's context.
  Used in the eval Sample Predictions prompt cell and `GoldEntryRowBody` (a row's tool list counts as used only when
  the caller passes `readsTools`: `GoldSetPanel` derives it with `projectReadsRowTools(project)` — task profile
  `tool_calling`, the same rule as the eval metrics — whether or not the row has a `tool_name`; callers that don't
  know (drift review, LLM preview) show "not used by this task"). `ToolCallMetricsPanel` `tcm-flags` line: "Per-row tool lists on X/Y rows (N distinct tools)" when
  `tool_flags.per_row`. "prompt rendering vN" label (renamed from "prompt format vN", P5b gap 5: it read like a variant name): Results history (only once a v2 result exists) and Compare cards.
  No new panel, tab or required step. Not placed: the adapter mapping preview (shows adapter output only).

## Eval sets (hardening Phase 3, 2026-11-19)

- No new tab or panel: `EvalSetsCard` sits inside the Results > Verdict run form ("Evaluate on test examples") under a
  closed `<details>` "More options: eval sets". The Dataset select lists eval sets in an "Eval sets" optgroup; picking
  one shows Eval-set version + Prompt format selects. Without eval sets the form is unchanged (the sets are fetched
  only when the form opens). Compare's "Evaluated on" picker appears only once an eval-set result exists.
- Labels (picked-data provenance): history cell "eval set <slug> v<N> (<variant>) "<name>" · R rows · content <sha8> ·
  prompt rendering vN", Compare card dataset line, bell Job title. Details: `agent-notes/evaluation.md` ("Eval sets").

## UI simplification pass — Advanced disclosure (phase 0, 2026-11-20)

- `components/shared/AdvancedSection.tsx` is the one "Advanced" disclosure for hiding controls most users never need.
  Collapsed by default, label "Advanced (N)" (`count`). Open / closed is remembered per page and section under
  localStorage `brewslm.advanced.<pageKey>.<id>` (every access in try/catch; it works with storage blocked; `pageKey` is
  read once at mount). The toggle follows the `DetailsDrawer` pattern (a native button with
  `aria-expanded={open ? 'true' : 'false'}`); design tokens only, no inline styles; `data-testid` defaults to
  `advanced-<id>` (`-toggle`, `-body`). Not used on any page yet; phases 1+ adopt it, no global switch.
- **Hidden fields keep sending today's values.** By default the children are NOT mounted while collapsed. A form whose
  Advanced fields feed a request must hold that state in the parent (controlled inputs) or pass `keepMounted` (children
  render inside a `hidden` wrapper and keep their own state). Never rely on an unmounted child to supply a default.
- **⌘K contract:** anything inside an AdvancedSection stays reachable from the palette. Give the section an `id`, register
  it in `utils/advancedSections.ts` (`ADVANCED_SECTION_IDS`), add a command in `utils/workspaceCommands.ts` whose path
  carries `?open=<id>`, and pass the page's `?open=` value as `openSection`: it opens the section for that visit (without
  changing the remembered state) and scrolls it into view; `openNonce` re-opens it after the user closed it. Old `?open=`
  / `?tab=` links keep working through `workspaceRoutes` redirects. `advancedSections.test.ts` fails when a registered id
  has no palette command.
- Plan and decisions: `ROADMAP-UI-SIMPLIFY.md` (local, gitignored). The counter that measures the pass is in
  `agent-notes/testing.md` ("Main-path counter").

## UI simplification pass — one Verdict page (phase 1, 2026-11-20)

- **Results › Verdict top = `components/evaluation/VerdictCard.tsx`** ("Is the fine-tune better than the base model?"),
  read-only over what is stored (no API change). Per evaluated set (newest result the run's own weights answered;
  another model's result only when the set has no own one, labelled "answered by model path X"): the score of record
  picked by scoring mode / key, never by value (`verdictScore.scoreOfRecord` + `SCORE_KEYS`, in the backend's lift
  headline order; strict tool-call rate when stored; full rule in `agent-notes/evaluation.md`), named by its key
  ("Span F1 (micro)", "Macro F1", "Token F1"…), a Wilson 95% interval only for a share-of-rows key with a known count,
  "N of M rows pass", "Partial: N of M rows scored" when the rows answered (`details.inference.samples`) < the set's
  rows, scorer-excluded rows on their own line, unjudged rows left out of the rate, and a slices hint when there is no
  `slice_breakdown`. Every line names run, set label (version, variant, rows, content hash, prompt rendering) and
  result id. Base vs fine-tuned: one `sft-lift-summary` call per set whose score is shown (built-in sets = one call;
  shared with Details through `createSharedLiftFetch`), the lift row on the score-of-record metric, rows better /
  worse with McNemar, "N rules regressed / N still failing · details" from `gate_status`, both runs and result ids,
  and **Open side by side** (`results?tab=compare&a=<baseline run>&b=<run>&set=<label>&eval_a=&eval_b=`; Compare
  pins those two results until the set picker changes). Rules line: PASS / FAIL / NOT MEASURED badges kept apart
  (failed rule ids listed, missing metrics named as "not a failure of the model", the measured seed run for a seed
  group; no report = "not checked yet", a failed fetch = "could not be loaded"), safety risk, and "details" which
  opens `verdict-details`. Under it the summary card (`verdictShowsLift`) reads "More on this run" without a second
  base-vs-fine-tuned headline. ScorecardPanel's "Inspect built-in checks" and the probe pack's "open run" link to
  their tab + section (`?tab=failures&open=failures-tools#probe-pack-panel`, `?tab=verdict&open=verdict-details`).
- **Moved, nothing removed** (each one click away + in ⌘K): score panels Failures → Verdict › Scores; rule report,
  SftLiftPanel ("Base vs fine-tuned", was "Did SFT help?"), scorecard, runtime, safety → Verdict › Details; probe pack,
  rule-set card (+ slices editor), planner, behavioral, active learning, reroute, drift, radar, answer-key review /
  workbench → Failures › Advanced (`failures-tools`). The LLM-judge side-by-side now renders on Failures only (it
  showed on every tab). `HowItsScoredCard` (decision 5): how each result was scored (mode, strict
  arguments, null as absent, prompt rendering, max new tokens / temperature); since phase 4 it also holds the
  project's scoring settings (see "phase 4" below).
- `?open=` on Results: `ProjectResultsPage.resolveResultsOpen` (`?open=` else `RESULTS_OPEN_BY_HASH`: the old
  `#probe-pack-panel` / `#reroute-recommendation-panel` / `#eval-pack-editor` anchors open `failures-tools`,
  `#experiment-scorecard` opens `verdict-details`) → `EvalPanel openSection`. Palette commands: "Scores", "Verdict
  details", "Failure tools" (`workspaceCommands` `RESULTS_ADVANCED`); ids in `advancedSections.ts`.
- O2 display: with a task scoring mode on the list, plain exact-match results without a scoring mode go under "Other
  stored metrics (string match)" in Metric History (`verdictScore.isStringMatchAside`) and the Verdict card names how
  many (no lift line for them); Compare moves its `exact_match` delta row aside only where it duplicates the mode's
  pass rate (`EXACT_MATCH_ALIAS_MODES`: tool call, structured plan, multiple options, remediation). O1: Compare cluster headings on a base vs fine-tuned
  pair read "Fails only with the fine-tune" / "Fixed by the fine-tune". Visible "Experiment" → "Run" on these screens.
- Tests: `VerdictCard.test.tsx` (fixtures shaped like stored results and `_compute_metric_lifts` rows; key order vs
  the backend list, span / finding / JSON EM 0 vs EM > 0 same metric, classification, plan / remediation, multi_option
  excluded rows without a false Partial, strict lift row, eval_a / eval_b link, gate_status counts, other-model result,
  seed run, gate fetch failure, aside without orphan lift line, HowItsScoredCard strict on / off / not recorded),
  `EvalPanel.workflows.test.tsx` (every moved panel mounts under its section, `?open=`, verdictShowsLift, probe "open
  run"), `EvalSummaryCard.test.tsx` ("More on this run"), `ProjectEvalComparePage.test.tsx` (O1 / O2, pin),
  `ProjectResultsPage.test.tsx`,
  `advancedSections.test.ts`. Tests that need a moved panel mount `workflow="verdict"` or open the section first.

## UI simplification pass — short "Score this run" form (phase 2, 2026-11-20)

- Results › Verdict: "+ Score this run" (`eval-score-run-open`, was "+ Evaluate on test examples") opens
  `components/evaluation/EvalRunForm.tsx`: What / Which model / Rows / Run on page one, everything else in its
  `AdvancedSection` `score-run-advanced` (keepMounted; the form is the parent that holds every field). "Base model" is a
  radio with the run's base model name (no typing into a path field); the raw "Model path" and the judge host / port
  live under Advanced. Detail, payload proof and prefills: `agent-notes/evaluation.md` ("Score this run").
- Eval sets left the form: Verdict › "Test sets" (`verdict-test-sets`, collapsed, mounts `EvalSetsCard` only while
  open and loads the sets then). The form's "Add an eval set" link opens it (`EvalPanel` `localOpen`).
- `?open=score-run-advanced` opens the form and its Advanced section (`EvalPanel` derives `showRunForm` from the
  request; the project-reset effect no longer closes it on first mount). Both ids are in `advancedSections.ts` with
  palette commands ("Score this run: advanced options", "Test sets").
- Picked-data provenance: the form names the run ("stores the score on run #N · name"), the rows line names the set and
  where its count came from ("Test examples: 7 rows (prepared split) · all rows prefilled (sets of up to 2,000 rows)",
  "your last choice", "partial: 100 of 2,500 rows", "matching the base result #N"), and "Also sent" names the hidden
  metric / tokens (+ source) / temperature (and the LLM judge's provider + model). Review fixes: rows / tokens match the
  stored base result for "Fine-tuned" (warning + "Match the base" when changed), "Scored as: <mode>", "Retry base
  only", per-set remembered tokens with "use suggestion", Run disabled while a remembered eval set loads; the Verdict
  lift line flags "Not like for like" (`evaluation.md`).
- Main path: Score this run is 2 counted actions (open, Run), was 5 (open, Max Samples, Max New Tokens, Temperature, Run
  Evaluation); see `agent-notes/testing.md`.
- Tests: `EvalRunForm.test.tsx` (payload equality vs `EvalRunForm.legacyBodies.json` — regenerate with
  `frontend/scripts/legacy-eval-capture/capture.sh` — prefills, base-result matching, Scored as, last used, blocked
  storage, baseline run, partial failure + retry), `EvalPanel.evalSets.test.tsx` (Test sets section, "Add an eval set",
  `?open=score-run-advanced`), `VerdictCard.test.tsx` / `ProjectEvalComparePage.test.tsx` (fixes A / B),
  `advancedSections.test.ts`.

## UI simplification pass — data import on step 1, Split trimmed (phase 3, 2026-11-20)

- Data › Sources leads with `components/data/OwnFilesSourcesCard.tsx` ("I already have train / validation / test
  files", `own-files-sources`): the same `PredefinedSplitCard` the Split step offers (that entry stays), wired by
  `ownFilesWiring.useOwnFilesWiring` with what the split form would send on mount (adapter preference + active prepared
  manifest: chat template and whether it is sent, adapter wiring, own-test-file guard). `ownFilesWiring.test.tsx` proves
  both placements post the same multipart body for five server states, with and without a column mapping.
- `PredefinedSplitCard`: one multi-file pick (`pdsplit-files`) assigns roles by file-name word
  (`roleFromFileName`: train / training, val / valid / validation / dev, test / testing, + set / split / data / digits;
  not `devops` / `contest`; two roles in one name = none), editable per file (`pdsplit-role-<i>`, "Not used"); two files
  on one role or no train file blocks Import. The per-role pickers (`pdsplit-train/val/test`), the leak choice, the
  chat template and the column mapping sit under its `AdvancedSection` `own-files-options` (`testId` `pdsplit-options`).
  A 409 leak offers "Drop the overlapping rows from train and import again" (`pdsplit-retry-drop`, resends with
  `drop_from_train`). Request fields unchanged.
- After an import on Sources, "Continue to Train" (`own-files-continue-train`; advances the pipeline like the step
  footers, then opens Train). Labels and Generate tabs carry an "optional" tag (`stage-step-optional-<tab>`).
- Split step: ratios + seed, keep-test / replace-own-test, Run Split (`split-run`) on top; adapter + task profile,
  presets, adapter / field-mapping JSON, domain-defaults checkbox, chat template, stratify / disjoint, Preview Effective
  Config (+ its panel) under `AdvancedSection` `split-advanced` (state stays in `DatasetPrepPanel`, so hidden values are
  still sent). Both ids are in `advancedSections.ts` with palette commands (`workspaceCommands` `DATA_ADVANCED`).
- Main path: Data is 2 counted actions (one pick of three files, Import), was 6 (step 5, toggle, three pickers,
  Import), and Data › Split is no longer a screen on the path; see `agent-notes/testing.md`.
- Tests: `PredefinedSplitCard.test.tsx` (roles by name, multi = three pickers body, role edits, G5 sentence, retry
  drop), `ownFilesWiring.test.tsx`, `ProjectDataPage.test.tsx`, `DatasetPrepPanel.hydrate.test.tsx`.

## UI simplification pass — Train in two clicks, scoring settings in one place (phase 4, 2026-11-20)

- Train › More options: "Common" (`TrainFormSection.group: 'common'`: Model and hyperparameters, LoRA and advanced
  training, Seeds) then `AdvancedSection` `train-advanced` (pageKey `train`) with every other section (Checks, Cloud
  burst, Training presets, Plan and gaps, registry, task type, pipeline presets, Adapter Studio, Tokenization).
  `?open=<advanced section>`, Fix (Checks), Browse (registry) and Change (task) open Advanced too; `?open=train-advanced`
  opens just the disclosure. TrainingPanel owns every field, so closed sections still send their values:
  `TrainingPanel.payload.test.tsx` snapshots five create bodies captured BEFORE the change (defaults, seeds + name,
  Common edits, defaults switch off in a closed Advanced section, prefilled from the latest run) — never update it
  to make a moved control pass.
- Train on a run that never started: when the newest run is a plain `pending` run (not started, no seed group, not a
  baseline, one seed), the form was prefilled from it and its create body is unchanged since (`prefillSnapshot`), with no
  run name and the same base model, the form says "Run #N · name was created with these settings and has not started"
  (`train-form-waiting-run`) and Train opens that run's confirmation (no second run). Any edit creates a run as before;
  Start on the row is unchanged.
- Scoring settings moved to Results › Verdict › `HowItsScoredCard` › `AdvancedSection` "Change how it's scored"
  (`how-scored-settings`, pageKey `results-verdict`): `ScoringSettingsSection` mounts (only while open) the same
  cards Train held — `ToolScoringConfigCard` (tool calling / structured plan, `effective_task_profile`),
  `StructuredPlanConfigCard`, `RemediationConfigCard`, `SqlExecutionConfigCard`, `AutoLiftSettingCard` — plus
  `NullAsAbsentProjectSetting` (the project switch; the Score this run form keeps the per-eval override and points
  here, `NullAsAbsentSetting showProjectToggle={false}`). Same endpoints and bodies. The card shows with no run
  selected. `ProjectTrainPage` redirects `?open=tool-scoring|structured-plan|remediation|sql-execution|auto-lift|null-as-absent`
  to `/results?open=how-scored-settings&setting=<id>` (scrolls to `how-scored-setting-<id>`). Both new ids are in
  `advancedSections.ts` with palette commands.
- Main path unchanged at Train (Train, Launch = 2); see `agent-notes/testing.md`.
- Tests: `TrainingPanel.payload.test.tsx`, `TrainForm.test.tsx`, `TrainingPanel.form.test.tsx` (waiting run),
  `ProjectTrainPage.test.tsx` (redirects), `ScoringSettingsSection.test.tsx`, `EvalPanel.nullAsAbsent.test.tsx`,
  `VerdictCard.test.tsx`; `src/test/setup.ts` now also clears `brewslm.advanced.*` between tests.
