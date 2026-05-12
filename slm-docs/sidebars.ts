import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/quickstart',
        'getting-started/first-project',
        'getting-started/model-family-guide',
        'getting-started/adapter-studio-examples',
      ],
    },
    {
      type: 'category',
      label: 'Concepts',
      items: [
        'concepts/architecture',
        'concepts/projects-and-artifacts',
        'concepts/pipeline-stages',
        'concepts/beginner-mode',
      ],
    },
    {
      type: 'category',
      label: 'Setup',
      items: [
        'setup/install',
        'setup/auth-and-sso',
        'setup/environment',
      ],
    },
    {
      type: 'category',
      label: 'Pipeline workflows',
      items: [
        'workflows/pipeline-overview',
        'workflows/data-ingestion',
        'workflows/newbie-autopilot',
        'workflows/training',
        'workflows/evaluation-and-remediation',
        'workflows/export-and-deployment',
      ],
    },
    {
      type: 'category',
      label: 'Deployment',
      items: [
        'deployment/plan',
        'deployment/smoke-and-promote',
        'deployment/telemetry',
        'deployment/drift-checks',
        'deployment/rollback-and-score',
      ],
    },
    {
      type: 'category',
      label: 'Observability',
      items: [
        'observability/run-events',
        'observability/timeline',
        'observability/failure-clusters',
        'observability/support-bundles',
      ],
    },
    {
      type: 'category',
      label: 'Extensions',
      items: [
        'extensions/contracts',
        'extensions/scaffold',
        'extensions/validate-and-reload',
        'extensions/extension-studio',
        'extensions/cli',
      ],
    },
    {
      type: 'category',
      label: 'Reliability',
      items: [
        'reliability/measured-vs-estimated',
        'reliability/common-blockers',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/cli',
        'reference/api-surface',
        'reference/model-compatibility-matrix',
        'reference/glossary',
      ],
    },
  ],
};

export default sidebars;
