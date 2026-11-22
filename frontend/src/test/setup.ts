import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

afterEach(() => {
  cleanup();
  // The "Score this run" form remembers last-used values per project (EvalRunForm); never across tests.
  try {
    for (const key of Object.keys(window.localStorage)) {
      if (key.startsWith('brewslm.scoreRun.')) window.localStorage.removeItem(key);
      // AdvancedSection remembers open / closed per page: a section a test opened must not start open in the next.
      if (key.startsWith('brewslm.advanced.')) window.localStorage.removeItem(key);
    }
    // The Train form keeps unsaved edits per project in sessionStorage (TrainingPanel draft): never across tests.
    for (const key of Object.keys(window.sessionStorage)) {
      if (key.startsWith('brewslm.trainDraft.')) window.sessionStorage.removeItem(key);
    }
  } catch {
    // storage unavailable
  }
});

Object.defineProperty(window, 'scrollTo', {
  value: vi.fn(),
  writable: true,
});
