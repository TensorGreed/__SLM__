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
  the Advanced toggle and the runs/workbench sub-tabs): Verdict = summary card
  (`hideTryBox`), run picker, Evaluate form, aggregate badge, SftLiftPanel,
  ScorecardPanel, inference card; Failures = EvalGaps, ProbePack, the rule-set
  card, the remediation planner, slices, behavioral table, clusters, active
  learning, reroute, drift queue, charts, schema banner, failure rows, plus
  collapsed `<details>` for AnswerKeyReviewPanel and GoldSetWorkbenchPanel;
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
  decision log; it was a floating drawer on every page until 2026-10-18); the
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
