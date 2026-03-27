import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'BrewSLM Docs',
  tagline: 'Beginner-friendly guides for building domain-specific SLMs end-to-end.',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'http://localhost',
  baseUrl: '/',

  onBrokenLinks: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: 'docs',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    navbar: {
      title: 'BrewSLM Docs',
      logo: {
        alt: 'BrewSLM Docs',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/docs/getting-started/quickstart',
          label: 'Quickstart',
          position: 'left',
        },
        {
          to: '/docs/reference/api-surface',
          label: 'API Map',
          position: 'left',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Start Here',
          items: [
            {
              label: 'Quickstart',
              to: '/docs/getting-started/quickstart',
            },
            {
              label: 'First Project',
              to: '/docs/getting-started/first-project',
            },
          ],
        },
        {
          title: 'Workflows',
          items: [
            {
              label: 'Training',
              to: '/docs/workflows/training',
            },
            {
              label: 'Evaluation',
              to: '/docs/workflows/evaluation-and-remediation',
            },
            {
              label: 'Export',
              to: '/docs/workflows/export-and-deployment',
            },
          ],
        },
        {
          title: 'Reference',
          items: [
            {
              label: 'CLI',
              to: '/docs/reference/cli',
            },
            {
              label: 'API Map',
              to: '/docs/reference/api-surface',
            },
          ],
        },
      ],
      copyright: `Copyright ${new Date().getFullYear()} BrewSLM Docs.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
