import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  fullyParallel: false,
  workers: 1,
  timeout: 60_000,
  use: { baseURL: 'http://127.0.0.1:9010', trace: 'retain-on-failure' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command:
      'cd .. && AGENTX_DATA_DIR=.agentx-e2e AGENTX_API_TOKEN= AGENTX_DATABASE_URL= AGENTX_STEPFUN_API_KEY= AGENTX_PLANNER_MODEL= AGENTX_PLANNER_BASE_URL= AGENTX_SAM2_MODEL_PATH= AGENTX_COSMOS_BACKEND=http AGENTX_COSMOS_MODEL_PATH= AGENTX_COSMOS_BASE_URL= AGENTX_WORKER_ENABLED=true uv run --no-sync agentx serve --port 9010',
    url: 'http://127.0.0.1:9010/api/health',
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
