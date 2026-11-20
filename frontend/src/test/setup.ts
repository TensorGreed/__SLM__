import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

afterEach(() => {
  cleanup();
  // The "Score this run" form remembers last-used values per project (EvalRunForm); never across tests.
  try {
    for (const key of Object.keys(window.localStorage)) {
      if (key.startsWith('brewslm.scoreRun.')) window.localStorage.removeItem(key);
    }
  } catch {
    // storage unavailable
  }
});

Object.defineProperty(window, 'scrollTo', {
  value: vi.fn(),
  writable: true,
});
