import type { ReactNode } from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';

import styles from './index.module.css';

interface TileProps {
  title: string;
  body: string;
  href: string;
}

const TILES: TileProps[] = [
  {
    title: 'Quickstart',
    body: 'Ten minutes from clone to a trained model. Backend + frontend + autopilot.',
    href: '/docs/getting-started/quickstart',
  },
  {
    title: 'Concepts',
    body: 'Architecture, projects, the 11 pipeline stages, beginner mode.',
    href: '/docs/concepts/architecture',
  },
  {
    title: 'Pipeline workflows',
    body: 'Per-stage UI / CLI / API for ingestion through export.',
    href: '/docs/workflows/pipeline-overview',
  },
  {
    title: 'Deployment',
    body: 'Plan, smoke, promote, telemetry, drift, rollback, deployability score.',
    href: '/docs/deployment/plan',
  },
  {
    title: 'Observability',
    body: 'Run timeline, failure clusters, support bundles, reason codes.',
    href: '/docs/observability/run-events',
  },
  {
    title: 'Extensions',
    body: 'Plugin contracts, scaffold generator, validate + reload, Extension Studio.',
    href: '/docs/extensions/contracts',
  },
];

function BrandMark() {
  return (
    <svg
      width="48"
      height="48"
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M 9 3.5 Q 7.5 6 9 8.5 Q 10.5 11 9 13.5" />
      <path d="M 15 3.5 Q 13.5 6 15 8.5 Q 16.5 11 15 13.5" />
      <path d="M 4 16 L 20 16" strokeWidth="1.6" />
    </svg>
  );
}

function Hero() {
  return (
    <header className={styles.hero}>
      <div className={styles.heroInner}>
        <BrandMark />
        <h1 className={styles.heroTitle}>BrewSLM Documentation</h1>
        <p className={styles.heroSubtitle}>
          A local-first platform for building, evaluating, and deploying
          domain-specific Small Language Models end-to-end. One UI, one CLI,
          one API.
        </p>
        <div className={styles.heroButtons}>
          <Link className={styles.buttonPrimary} to="/docs/getting-started/quickstart">
            Start with Quickstart →
          </Link>
          <Link className={styles.buttonSecondary} to="/docs/concepts/architecture">
            Read concepts first
          </Link>
        </div>
      </div>
    </header>
  );
}

function Tiles() {
  return (
    <section className={styles.tiles}>
      <div className={styles.tilesInner}>
        {TILES.map((tile) => (
          <Link key={tile.href} to={tile.href} className={styles.tile}>
            <h3 className={styles.tileTitle}>{tile.title}</h3>
            <p className={styles.tileBody}>{tile.body}</p>
            <span className={styles.tileArrow}>→</span>
          </Link>
        ))}
      </div>
    </section>
  );
}

function ThreeSurfaces() {
  return (
    <section className={styles.surfaces}>
      <div className={styles.surfacesInner}>
        <h2 className={styles.surfacesTitle}>Three surfaces, one source of truth</h2>
        <p className={styles.surfacesIntro}>
          Every feature in BrewSLM is reachable from the web UI, the
          <code> brewslm </code> CLI, and the HTTP API. Most doc pages show all
          three for the same operation.
        </p>
        <div className={styles.surfacesGrid}>
          <div className={styles.surfaceCard}>
            <div className={styles.surfaceLabel}>UI</div>
            <p>Click through. See live state. Pick defaults. Watch loss curves. Cmd-K to navigate anywhere.</p>
          </div>
          <div className={styles.surfaceCard}>
            <div className={styles.surfaceLabel}>CLI</div>
            <p>Script it. Gate it in CI. Reproduce a teammate's run. Pipe to <code>jq</code>.</p>
          </div>
          <div className={styles.surfaceCard}>
            <div className={styles.surfaceLabel}>API</div>
            <p>Embed in your tools. Build integrations. Trigger from a webhook. Live Swagger at <code>/api/docs</code>.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout
      title="BrewSLM Documentation"
      description="Comprehensive product guide for BrewSLM: data ingestion, fine-tuning, evaluation, deployment, observability, and extensions."
    >
      <Hero />
      <Tiles />
      <ThreeSurfaces />
    </Layout>
  );
}
