import { expect, test } from '@playwright/test';

// Chromium's synthetic camera: an animated test pattern, no real device or permission prompt.
test.use({
  launchOptions: {
    args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
  },
  permissions: ['camera'],
});

test('a live camera recording can be asked while memory follows it, then sealed', async ({
  page,
  request,
}) => {
  const browserErrors: string[] = [];
  page.on('pageerror', (error) => browserErrors.push(error.message));
  await page.goto('/');
  await page.getByRole('button', { name: 'Live camera', exact: true }).first().click();
  await expect(page.locator('.tag.live')).toBeVisible();
  await expect(page.getByLabel('Live camera preview')).toBeVisible();
  await expect(page.locator('.live-controls')).toContainText(/[1-9]\d* frames sent/, {
    timeout: 15_000,
  });

  await expect(page.getByRole('button', { name: 'Register object', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Register object', exact: true }).click();
  const modal = page.getByRole('dialog', { name: 'Register an object' });
  await modal.getByRole('textbox', { name: 'Object name' }).fill('Test pattern');
  await modal.getByRole('button', { name: 'Enter bounding coordinates instead' }).click();
  await modal.getByText('Adjust normalized bounds with keyboard').click();
  // The synthetic pattern is mostly flat colour; a wide box keeps enough texture to track.
  await modal.getByRole('spinbutton', { name: 'Box left' }).fill('0.05');
  await modal.getByRole('spinbutton', { name: 'Box top' }).fill('0.05');
  await modal.getByRole('spinbutton', { name: 'Box right' }).fill('0.95');
  await modal.getByRole('spinbutton', { name: 'Box bottom' }).fill('0.95');
  await modal.getByRole('button', { name: 'Save reference' }).click();
  await expect(modal).not.toBeVisible();

  await page.getByRole('button', { name: 'Build memory', exact: true }).click();
  await expect(page.locator('.progress-meta')).toContainText('Following the live camera', {
    timeout: 15_000,
  });
  await expect(page.getByText('Live memory')).toBeVisible();
  await expect(page.locator('.live-feed')).toContainText('Test pattern', { timeout: 15_000 });
  // The feed keeps only the newest five events, and the synthetic pattern flickers, so the first
  // sighting is checked in the recorded history rather than in the feed.
  const recording = (await (await request.get('/api/v1/videos')).json()).find(
    (v: { live_status: string | null }) => v.live_status === 'recording',
  );
  await expect
    .poll(async () =>
      (await (await request.get(`/api/v1/runs/${recording.runs[0].id}/events`)).json()).map(
        (e: { kind: string }) => e.kind,
      ),
    )
    .toContain('appeared');
  const composer = page.getByRole('textbox', { name: 'Ask your memory' });
  await expect(composer).toBeEnabled();
  await composer.fill('Where is Test pattern?');
  await page.getByRole('button', { name: 'Send question' }).click();
  await expect(page.locator('.answer-text')).toHaveCount(1, { timeout: 20_000 });
  await expect(page.locator('.warning-note').first()).toContainText('live recording');
  await page.screenshot({ path: 'test-results/live-following.png', fullPage: true });

  await page.getByRole('button', { name: 'Stop & seal' }).click();
  await expect(page.getByText('Live capture · sealed')).toBeVisible({ timeout: 30_000 });
  const videos = await (await request.get('/api/v1/videos')).json();
  const live = videos.find((v: { live_status: string | null }) => v.live_status === 'sealed');
  expect(live.sha256).toMatch(/^[0-9a-f]{64}$/);
  await expect
    .poll(async () => (await (await request.get(`/api/v1/runs/${live.runs[0].id}`)).json()).status)
    .toBe('complete');
  await expect(page.getByRole('slider', { name: 'Query cutoff' })).toBeVisible({
    timeout: 15_000,
  });
  expect(browserErrors).toEqual([]);
});
