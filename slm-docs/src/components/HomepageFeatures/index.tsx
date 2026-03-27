import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';

import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Novice-First Onboarding',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Follow practical guides that assume you are learning by building. Start from zero and ship a working
        domain model iteration safely.
      </>
    ),
  },
  {
    title: 'End-to-End Workflow',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Learn each stage of the BrewSLM pipeline: ingestion, training, evaluation, remediation, export, and
        deployment checks.
      </>
    ),
  },
  {
    title: 'Reliability and Operations',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Understand measured vs estimated metrics, strict mode behavior, and common blockers so your runs are
        reproducible and trustworthy.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
