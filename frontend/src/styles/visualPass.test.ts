/**
 * Visual-pass lock-in (Epic U phase U7).
 *
 * The pages and panels the workspace IA touched carry no inline
 * ``style={{ … }}`` objects and no emoji glyphs (icons are lucide), and no
 * component or stylesheet other than ``index.css`` writes a colour literal —
 * every colour is a token (charts: ``--chart-*``), so the dark theme covers it. No file anywhere
 * draws a disclosure triangle glyph — chevrons are lucide ``ChevronRight`` / ``ChevronDown``. Add a file to the lists when it
 * joins the clean set; never remove one.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { describe, expect, it } from 'vitest';

const SRC = join(__dirname, '..');

/** Components and pages that must stay free of inline styles and emoji. */
export const CLEAN_FILES = [
    'components/annotation/SpanLabeler.tsx',
    'components/autopilot/DecisionLogPanel.tsx',
    'components/coach/CoachNotices.tsx',
    'components/coach/CoachSuggestion.tsx',
    'components/coach/StepHelp.tsx',
    'components/dashboard/DemoProjectTiles.tsx',
    'components/data/CleaningPanel.tsx',
    'components/data/DataHealthReportPanel.tsx',
    'components/data/DataStudioReviewQueuePanel.tsx',
    'components/data/DatasetImportWizard.tsx',
    'components/data/DatasetPrepPanel.tsx',
    'components/data/DatasetVersionPanel.tsx',
    'components/data/DocumentSourceImportCard.tsx',
    'components/data/GoldSetPanel.tsx',
    'components/data/PredefinedSplitCard.tsx',
    'components/data/ProvenancePanel.tsx',
    'components/data/IngestionPanel.tsx',
    'components/data/LabelTaxonomyCard.tsx',
    'components/data/LabellingConventionsCard.tsx',
    'components/data/OutputSchemaEditor.tsx',
    'components/data/SyntheticPanel.tsx',
    'components/data/TaskShapeCard.tsx',
    'components/evaluation/EvalGapsPanel.tsx',
    'components/evaluation/EvalPackScaffoldPanel.tsx',
    'components/evaluation/EvalPanel.tsx',
    'components/evaluation/ExternalBenchmarksPanel.tsx',
    'components/evaluation/FailureClustersPanel.tsx',
    'components/evaluation/FieldMatchMetricsPanel.tsx',
    'components/evaluation/RemediationMetricsPanel.tsx',
    'components/evaluation/RerouteRecommendationPanel.tsx',
    'components/evaluation/runSelection.ts',
    'components/evaluation/ScorecardPanel.tsx',
    'components/evaluation/SeedShipTable.tsx',
    'components/evaluation/SftLiftPanel.tsx',
    'components/domain/DomainPackContentList.tsx',
    'components/domain/DomainPackImportPanel.tsx',
    'components/evaluation/UserProbeEditor.tsx',
    'components/export/ExportPanel.tsx',
    'components/guide/QuickstartCard.tsx',
    'components/layout/CommandPalette.tsx',
    'components/layout/TopBar.tsx',
    'components/layout/ProjectSidebar.tsx',
    'components/observability/TimelineTree.tsx',
    'components/projects/ProjectTemplateGallery.tsx',
    'components/shared/DetailsDrawer.tsx',
    'components/shared/EmptyState.tsx',
    'components/shared/GenerationModelPicker.tsx',
    'components/shared/HealthCheckButton.tsx',
    'components/shared/LoadingState.tsx',
    'components/shared/NoRecipeEmptyState.tsx',
    'components/shared/Skeleton.tsx',
    'components/shared/StepFooter.tsx',
    'components/shared/Toast.tsx',
    'components/training/DatasetFitCard.tsx',
    'components/training/ExperimentCompare.tsx',
    'components/data/CodeRemediationParamsForm.tsx',
    'components/training/RemediationConfigCard.tsx',
    'components/training/TokenLengthDistributionPanel.tsx',
    'components/training/TokenizationPanel.tsx',
    'components/training/TrainForm.tsx',
    'components/training/TrainingConfigGapsPanel.tsx',
    'components/training/TrainingPanel.tsx',
    'components/training/TrainingRunsList.tsx',
    'components/video/TabVideoLink.tsx',
    'data/tabVideos.ts',
    'pages/ProjectAnnotatePage.tsx',
    'pages/ProjectAutopilotPage.tsx',
    'pages/ProjectDataPage.tsx',
    'pages/ProjectDeploymentsPage.tsx',
    'pages/ProjectEvalComparePage.tsx',
    'pages/ProjectListPage.tsx',
    'pages/ProjectManifestPage.tsx',
    'pages/ProjectOverviewPage.tsx',
    'pages/ProjectRecipePickerPage.tsx',
    'pages/ProjectResultsPage.tsx',
    'pages/ProjectSettingsPage.tsx',
    'pages/ProjectShipPage.tsx',
    'pages/ProjectTrainPage.tsx',
    'pages/ProjectWorkspaceLayout.tsx',
];

const INLINE_STYLE = /\bstyle=\{/;
const EMOJI = /[\p{Extended_Pictographic}✓✔✕✖✗★☆⏳⭐]/u;
const HEX_COLOUR = /#[0-9a-fA-F]{3,8}\b/;
/** A colour literal in a component: hex, rgb()/rgba(), hsl(). */
const COLOUR_LITERAL = /#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(/;
/** Disclosure / arrow triangles (U+25B2–U+25C5) — use lucide chevrons and arrows. */
const DISCLOSURE_GLYPH = /[\u25B2-\u25C5]/;

function walkFiles(dir: string, extensions: string[], out: string[] = []): string[] {
    for (const entry of readdirSync(dir)) {
        const full = join(dir, entry);
        if (statSync(full).isDirectory()) walkFiles(full, extensions, out);
        else if (extensions.some((ext) => entry.endsWith(ext))) out.push(full);
    }
    return out;
}

const walkCss = (dir: string): string[] => walkFiles(dir, ['.css']);

function offendingLines(text: string, pattern: RegExp): string[] {
    return text
        .split('\n')
        .map((line, i) => (pattern.test(line) ? `${i + 1}: ${line.trim().slice(0, 100)}` : null))
        .filter((line): line is string => line !== null);
}

describe('visual pass (U7)', () => {
    it('the cleaned components carry no inline styles', () => {
        const report: Record<string, string[]> = {};
        for (const file of CLEAN_FILES) {
            const lines = offendingLines(readFileSync(join(SRC, file), 'utf8'), INLINE_STYLE);
            if (lines.length) report[file] = lines;
        }
        expect(report).toEqual({});
    });

    it('the cleaned components carry no emoji glyphs', () => {
        const report: Record<string, string[]> = {};
        for (const file of CLEAN_FILES) {
            const lines = offendingLines(readFileSync(join(SRC, file), 'utf8'), EMOJI);
            if (lines.length) report[file] = lines;
        }
        expect(report).toEqual({});
    });

    it('the cleaned components carry no colour literal (charts use the --chart-* tokens)', () => {
        const report: Record<string, string[]> = {};
        for (const file of CLEAN_FILES) {
            const lines = offendingLines(readFileSync(join(SRC, file), 'utf8'), COLOUR_LITERAL);
            if (lines.length) report[file] = lines;
        }
        expect(report).toEqual({});
    });

    it('only index.css writes a hex colour; every other stylesheet uses tokens', () => {
        const report: Record<string, string[]> = {};
        for (const file of walkCss(SRC)) {
            const rel = relative(SRC, file);
            if (rel === 'index.css') continue;
            const lines = offendingLines(readFileSync(file, 'utf8'), HEX_COLOUR);
            if (lines.length) report[rel] = lines;
        }
        expect(report).toEqual({});
    });

    it('no component or stylesheet draws a disclosure glyph (chevrons are lucide)', () => {
        const report: Record<string, string[]> = {};
        for (const file of walkFiles(SRC, ['.tsx', '.ts', '.css'])) {
            const rel = relative(SRC, file);
            const lines = offendingLines(readFileSync(file, 'utf8'), DISCLOSURE_GLYPH);
            if (lines.length) report[rel] = lines;
        }
        expect(report).toEqual({});
    });
});
