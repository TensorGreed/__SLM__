import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/demo-recordings',
  timeout: 60_000,
  expect: {
    timeout: 10_000,
  },
  use: {
    // DEMO_BASE_URL points a recording at a scratch stack (docs-demo/scripts/ffmpeg/scratch-stack.sh).
    baseURL: process.env.DEMO_BASE_URL || 'http://localhost:5173',
    viewport: { width: 1440, height: 900 },
    video: {
      mode: 'on',
      size: { width: 1440, height: 900 },
    },
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
    launchOptions: {
      slowMo: 250,
    },
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1440, height: 900 },
      },
    },
  ],
});

