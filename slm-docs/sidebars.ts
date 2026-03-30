import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/quickstart',
        'getting-started/first-project',
        'getting-started/model-family-guide',
      ],
    },
    {
      type: 'category',
      label: 'Workflows',
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
      label: 'Reliability',
      items: ['reliability/measured-vs-estimated', 'reliability/common-blockers'],
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
