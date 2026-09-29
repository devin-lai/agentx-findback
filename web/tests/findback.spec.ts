import { expect, test } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

test('review points align with source pixels and remain model interpretations', async ({
  page,
  request,
}) => {
  const video = await (await request.post('/api/v1/demo')).json();
  const run = await (await request.post(`/api/v1/videos/${video.id}/runs`, { data: {} })).json();
  await expect
    .poll(async () => (await (await request.get(`/api/v1/runs/${run.id}`)).json()).status, {
      timeout: 30_000,
    })
    .toBe('complete');
  await page.route('**/review-fixture-portrait.svg', (route) =>
    route.fulfill({
      contentType: 'image/svg+xml',
      body: '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="400"><rect width="200" height="400" fill="#e8eee9"/><circle cx="150" cy="100" r="10" fill="#375e43"/></svg>',
    }),
  );
  await page.route('**/runs/*/reviews', (route) =>
    route.fulfill({
      json: [
        {
          id: 'browser-review',
          question: 'Inspect this software fixture.',
          mode: 'grounded',
          target: 'Test target',
          summary: 'A possible match in the first frame.',
          uncertainty: 'This is a mocked browser contract, not a model accuracy result.',
          notice: 'Model interpretation; original evidence remains authoritative.',
          as_of_ms: 7000,
          model: 'test-provider',
          elapsed_seconds: 1,
          authoritative: false,
          evidence_frame_ids: [0],
          skills: [],
          provenance: { backend: 'http', device: 'mock', attempts: 1, prompt_sha256: 'test' },
          frames: [0, 1].map((id) => ({
            id,
            at_ms: id * 7000,
            sha256: 'test',
            frame_url: '/review-fixture-portrait.svg',
          })),
          frame_reports: [
            {
              id: 0,
              at_ms: 0,
              present: true,
              x: 0.75,
              y: 0.25,
              zone: 'right',
              zone_name: 'Right area',
              note: 'Possible match',
            },
            {
              id: 1,
              at_ms: 7000,
              present: false,
              x: null,
              y: null,
              zone: null,
              zone_name: null,
              note: 'No match',
            },
          ],
        },
      ],
    }),
  );
  await page.goto('/');
  const review = page.locator('.review-card');
  await expect(review).toContainText('Model interpretation:');
  await expect(review).toContainText('Model: Right area');
  await expect(review).toContainText('No model match');
  await expect(review.locator('.review-point')).toHaveCount(1);
  const frame = review.getByRole('button', { name: 'Replay review frame 0 at 00:00.0' });
  await expect
    .poll(() => frame.locator('img').evaluate((img: HTMLImageElement) => img.naturalHeight))
    .toBe(400);
  const img = await frame.locator('img').boundingBox();
  const point = await frame.locator('.review-point').boundingBox();
  expect(Math.abs((point!.x + point!.width / 2 - img!.x) / img!.width - 0.75)).toBeLessThan(0.01);
  expect(Math.abs((point!.y + point!.height / 2 - img!.y) / img!.height - 0.25)).toBeLessThan(0.01);
  await review.screenshot({ path: 'test-results/review-model-points.png' });
  await review.getByRole('button', { name: 'Replay review frame 1 at 00:07.0' }).click();
  await expect
    .poll(() => page.locator('video').evaluate((v: HTMLVideoElement) => v.currentTime))
    .toBeGreaterThanOrEqual(7);
});

test('an agent-created review appears with its answer without reloading', async ({
  page,
  request,
}) => {
  for (const video of await (await request.get('/api/v1/videos')).json()) {
    await request.delete(`/api/v1/videos/${video.id}`);
  }
  const video = await (await request.post('/api/v1/demo')).json();
  const run = await (await request.post(`/api/v1/videos/${video.id}/runs`, { data: {} })).json();
  await expect
    .poll(async () => (await (await request.get(`/api/v1/runs/${run.id}`)).json()).status)
    .toBe('complete');

  const review = {
    id: 'agent-created-review',
    question: 'Inspect the toolkit at the cutoff.',
    mode: 'freeform',
    target: null,
    summary: 'New agent review from the generated fixture.',
    uncertainty: 'This interpretation is advisory.',
    notice: 'Model interpretation; saved memory remains authoritative.',
    as_of_ms: 7000,
    model: 'test-provider',
    elapsed_seconds: 1,
    authoritative: false,
    evidence_frame_ids: [],
    frames: [],
    frame_reports: [],
    skills: [],
    provenance: { backend: 'http', device: 'mock', attempts: 1, prompt_sha256: 'test' },
  };
  let reviewReady = false;
  let reviewReads = 0;
  await page.route(`**/api/v1/runs/${run.id}/reviews`, async (route) => {
    reviewReads += 1;
    await route.fulfill({ json: reviewReady ? [review] : [] });
  });
  await page.route(`**/api/v1/runs/${run.id}/questions`, async (route) => {
    if (route.request().method() !== 'POST') return route.continue();
    reviewReady = true;
    await route.fulfill({
      json: {
        id: 'agent-answer',
        run_id: run.id,
        question: 'Where is Red toolkit?',
        answer: 'The model review is advisory; the current position is uncertain.',
        as_of_ms: 7000,
        intent: 'location',
        states: [],
        events: [],
        evidence: [],
        skills: [],
        tools: [{ name: 'review_frames', result: 'review agent-cr · 0 cited frames' }],
        planner: 'agent',
        planner_model: 'test-provider',
        warnings: [],
        elapsed_seconds: 1,
      },
    });
  });
  await page.goto('/');
  await expect.poll(() => reviewReads).toBeGreaterThan(0);
  await expect(page.locator('.review-card')).toHaveCount(0);
  await page
    .getByRole('textbox', { name: 'Ask your memory', exact: true })
    .fill('Where is Red toolkit?');
  await page.getByRole('button', { name: 'Send question' }).click();
  await expect(page.locator('.answer-text')).toContainText('current position is uncertain');
  await expect(page.locator('.review-card')).toContainText(
    'New agent review from the generated fixture.',
  );
  await expect(page.locator('.review-card')).toContainText('Model interpretation:');
  expect(reviewReads).toBeGreaterThan(1);
});

test('sample → indexing → historical uncertainty → evidence → persistence', async ({
  page,
  request,
}) => {
  for (const video of await (await request.get('/api/v1/videos')).json()) {
    await request.delete(`/api/v1/videos/${video.id}`);
  }
  const browserErrors: string[] = [];
  page.on('pageerror', (error) => browserErrors.push(error.message));
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Give your footage a memory.' })).toBeVisible();
  await page.screenshot({ path: 'test-results/empty-workspace.png', fullPage: true });
  await page.getByRole('button', { name: 'Try controlled sample' }).click();
  await expect(page.getByText('120 observations', { exact: false })).toBeVisible({
    timeout: 30_000,
  });
  await expect(page.getByText('Generated software fixture', { exact: false })).toBeVisible();
  const cutoff = page.getByRole('slider', { name: 'Query cutoff' });
  await cutoff.fill('7000');
  await cutoff.dispatchEvent('change');
  await expect(page.locator('.memory-row').filter({ hasText: 'Red toolkit' })).toContainText(
    'Last seen',
  );
  await page
    .getByRole('textbox', { name: 'Ask your memory', exact: true })
    .fill('Where is Red toolkit?');
  await page.getByRole('button', { name: 'Send question' }).click();
  await expect(page.getByLabel('State of Red toolkit at cutoff')).toContainText(
    'No supported current position',
  );
  await expect(page.getByLabel('State of Red toolkit at cutoff')).toContainText(
    'Last confirmed in source · 00:05.8',
  );
  await expect(page.locator('.answer-text')).toContainText(
    'position at the requested time is not confirmed',
  );
  await expect(page.locator('.answer-text')).toContainText('00:05.8');
  await page.getByRole('button', { name: 'Inspect cutoff frame' }).click();
  await expect
    .poll(() => page.locator('video').evaluate((v: HTMLVideoElement) => v.currentTime))
    .toBeGreaterThanOrEqual(6.9);
  await page.getByRole('button', { name: 'Replay evidence at 00:05.8' }).click();
  await expect
    .poll(() => page.locator('video').evaluate((v: HTMLVideoElement) => v.currentTime))
    .toBeGreaterThanOrEqual(4.8);
  await page.getByText('Source and time checked').click();
  await expect(page.locator('.trace-content')).toContainText('retrieve-object-history');
  await page.getByRole('button', { name: /Event history/ }).click();
  await expect(page.locator('.event-list')).toContainText('Visual contact lost');
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/memory-evidence.png', fullPage: true });
  await page.reload();
  await expect(page.locator('.answer-text')).toContainText('not confirmed');
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('heading', { name: 'Give your footage a memory.' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  const mobilePanels = await page.evaluate(() => {
    const boxes = [
      '.footage-panel',
      '.registration-panel',
      '.assistant-panel',
      '.memory-panel',
    ].map((selector) => document.querySelector(selector)!.getBoundingClientRect());
    return boxes.map(({ top, width }) => ({ top, width }));
  });
  expect(mobilePanels.map(({ top }) => top)).toEqual(
    [...mobilePanels.map(({ top }) => top)].sort((a, b) => a - b),
  );
  expect(mobilePanels.every(({ width }) => width > 350)).toBe(true);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/mobile-workspace.png', fullPage: true });
  await page
    .getByRole('textbox', { name: 'Ask your memory', exact: true })
    .fill('Where is Blue remote?');
  await page.getByRole('button', { name: 'Send question' }).click();
  await expect(page.locator('.answer-text')).toHaveCount(1);
  await page.getByRole('button', { name: 'Show 1 earlier answer' }).click();
  await expect(page.locator('.answer-text')).toHaveCount(2);
  await page.getByRole('button', { name: 'Hide 1 earlier answer' }).click();
  await expect(page.locator('.answer-text')).toHaveCount(1);
  expect(browserErrors).toEqual([]);
});

test('citation box stays on the source pixels for portrait footage', async ({ page, request }) => {
  for (const video of await (await request.get('/api/v1/videos')).json()) {
    await request.delete(`/api/v1/videos/${video.id}`);
  }
  const video = await (await request.post('/api/v1/demo')).json();
  const run = await (await request.post(`/api/v1/videos/${video.id}/runs`, { data: {} })).json();
  await expect
    .poll(async () => (await (await request.get(`/api/v1/runs/${run.id}`)).json()).status)
    .toBe('complete');
  const answer = await (
    await request.post(`/api/v1/runs/${run.id}/questions`, {
      data: { text: 'Where is Red toolkit?', at_ms: 7000, use_provider: false },
    })
  ).json();
  const expected = answer.evidence[0].box;
  await page.route('**/api/v1/observations/*/frame', (route) =>
    route.fulfill({
      contentType: 'image/svg+xml',
      body: '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="400"><rect width="200" height="400" fill="#243629"/></svg>',
    }),
  );
  await page.goto('/');
  const card = page.locator('.evidence-card').first();
  await expect(card).toBeVisible();
  await expect
    .poll(() => card.locator('img').evaluate((img: HTMLImageElement) => img.naturalHeight))
    .toBe(400);
  const image = await card.locator('img').boundingBox();
  const box = await card.locator('.evidence-box').boundingBox();
  expect(image).not.toBeNull();
  expect(box).not.toBeNull();
  expect(image!.height / image!.width).toBeGreaterThan(1.9);
  expect(Math.abs((box!.x - image!.x) / image!.width - expected.x1)).toBeLessThan(0.02);
  expect(Math.abs((box!.y - image!.y) / image!.height - expected.y1)).toBeLessThan(0.02);
  expect(Math.abs((box!.x + box!.width - image!.x) / image!.width - expected.x2)).toBeLessThan(
    0.02,
  );
  expect(Math.abs((box!.y + box!.height - image!.y) / image!.height - expected.y2)).toBeLessThan(
    0.02,
  );
  await card.screenshot({ path: 'test-results/evidence-portrait.png' });
});

test('upload and registration support keyboard coordinates', async ({ page, request }) => {
  const sample = await (await request.post('/api/v1/demo')).json();
  const media = await request.get(sample.media_url);
  await page.goto('/');
  await page.getByLabel('Upload video file').setInputFiles({
    name: 'Uploaded desk.mp4',
    mimeType: 'video/mp4',
    buffer: await media.body(),
  });
  await expect(page.getByRole('heading', { name: 'Uploaded desk', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Register object', exact: true }).click();
  const modal = page.getByRole('dialog', { name: 'Register an object' });
  await expect(modal).toBeVisible();
  await modal.getByRole('textbox', { name: 'Object name' }).fill('Keyboard reference');
  await modal.getByRole('button', { name: 'Enter bounding coordinates instead' }).click();
  await modal.getByText('Adjust normalized bounds with keyboard').click();
  await modal.getByRole('spinbutton', { name: 'Box left' }).fill('0.53125');
  await modal.getByRole('spinbutton', { name: 'Box top' }).fill('0.65278');
  await modal.getByRole('spinbutton', { name: 'Box right' }).fill('0.67188');
  await modal.getByRole('spinbutton', { name: 'Box bottom' }).fill('0.83056');
  await modal.getByRole('button', { name: 'Save reference' }).click();
  await expect(modal).not.toBeVisible();
  await expect(
    page.locator('.object-chip').filter({ hasText: 'Keyboard reference' }),
  ).toBeVisible();
});

test('memory versions preserve their own answers, regions and object inventory', async ({
  page,
  request,
}) => {
  const video = await (await request.post('/api/v1/demo')).json();
  async function index() {
    const run = await (
      await request.post(`/api/v1/videos/${video.id}/runs`, {
        data: { backend: 'reference', sample_fps: 5 },
      })
    ).json();
    await expect
      .poll(async () => (await (await request.get(`/api/v1/runs/${run.id}`)).json()).status)
      .toBe('complete');
    return run;
  }
  const first = await index();
  await request.post(`/api/v1/runs/${first.id}/questions`, {
    data: { text: 'Where is Red toolkit?', at_ms: 2000, use_provider: false },
  });
  const regions = video.regions.map((region: { id: string; name: string }) => ({
    ...region,
    name: region.id === 'left' ? 'New region name' : region.name,
  }));
  const renamed = await request.put(`/api/v1/videos/${video.id}/regions`, { data: regions });
  expect(renamed.ok()).toBeTruthy();
  const registered = await request.post(`/api/v1/videos/${video.id}/objects`, {
    data: { name: 'Later reference', label: 'custom', at_ms: 1000, box: video.objects[1].box },
  });
  expect(registered.ok()).toBeTruthy();
  const second = await index();
  await page.goto('/');
  await page.getByRole('combobox', { name: 'Memory version' }).selectOption(first.id);
  await expect(page.locator('.answer-text')).toContainText('Left area');
  const scope = page.getByRole('combobox', { name: 'Query scope' });
  await expect(scope.locator('option')).toHaveCount(3);
  await scope.selectOption(video.objects[0].id);
  await page.getByRole('combobox', { name: 'Memory version' }).selectOption(second.id);
  await expect(page.locator('.answer-text')).toHaveCount(0);
  await expect(scope).toHaveValue('');
  await expect(scope.locator('option')).toHaveCount(4);
  const cutoff = page.getByRole('slider', { name: 'Query cutoff' });
  await cutoff.fill('2000');
  await cutoff.dispatchEvent('change');
  await expect(page.locator('.memory-row').filter({ hasText: 'Red toolkit' })).toContainText(
    'New region name',
  );
  await page.getByRole('combobox', { name: 'Memory version' }).selectOption(first.id);
  await expect(page.locator('.memory-row').filter({ hasText: 'Red toolkit' })).toContainText(
    'Left area',
  );
  await expect(cutoff).toHaveValue('2000');
  await page.screenshot({ path: 'test-results/memory-versions.png', fullPage: true });
});

test('a user can stop indexing and retain the stopped memory version', async ({
  page,
  request,
}) => {
  const video = await (await request.post('/api/v1/demo')).json();
  for (let index = 0; index < 8; index++) {
    const response = await request.post(`/api/v1/videos/${video.id}/objects`, {
      data: {
        name: `Reference ${index}`,
        label: 'custom',
        at_ms: 0,
        box: video.objects[index % 2].box,
      },
    });
    expect(response.ok()).toBeTruthy();
  }
  await page.goto('/');
  await page.getByRole('button', { name: 'Build memory', exact: true }).click();
  await page.getByRole('button', { name: 'Stop indexing', exact: true }).click();
  await expect(page.getByRole('status')).toContainText('Indexing stopped', { timeout: 15_000 });
  await expect(page.getByRole('button', { name: 'Rebuild memory', exact: true })).toBeEnabled();
  const result = await (await request.get(`/api/v1/videos/${video.id}`)).json();
  expect(result.runs[0].status).toBe('cancelled');
  await page.reload();
  await expect(page.getByRole('status')).toContainText('Indexing stopped');
});

test('saved answers download a verifiable report that opens offline', async ({
  page,
  request,
  browser,
}, testInfo) => {
  const video = await (await request.post('/api/v1/demo')).json();
  const run = await (await request.post(`/api/v1/videos/${video.id}/runs`, { data: {} })).json();
  await expect
    .poll(async () => (await (await request.get(`/api/v1/runs/${run.id}`)).json()).status, {
      timeout: 30_000,
    })
    .toBe('complete');
  await request.post(`/api/v1/runs/${run.id}/questions`, {
    data: { text: 'Show the history of Red toolkit', at_ms: 7000, use_provider: false },
  });
  await page.goto('/');
  const save = page.getByRole('button', { name: 'Save evidence report' });
  await page.route('**/questions/*/bundle', (route) =>
    route.fulfill({ status: 413, json: { detail: 'Ask with an earlier cutoff.' } }),
  );
  await save.click();
  await expect(page.getByRole('alert')).toContainText('Ask with an earlier cutoff.');
  await page.unroute('**/questions/*/bundle');
  const downloaded = page.waitForEvent('download');
  await save.click();
  const download = await downloaded;
  expect(download.suggestedFilename()).toMatch(/^findback-evidence-.*\.zip$/);
  const archive = testInfo.outputPath('evidence.zip');
  const folder = testInfo.outputPath('evidence');
  await download.saveAs(archive);
  await mkdir(folder, { recursive: true });
  const python = resolve('../.venv/bin/python');
  execFileSync(python, ['-m', 'zipfile', '-e', archive, folder]);
  const verified = JSON.parse(
    execFileSync(python, ['-I', `${folder}/verify.py`], { encoding: 'utf8' }),
  );
  expect(verified.status).toBe('verified');
  expect(verified.frames).toBeGreaterThan(1);
  expect(verified.cutoff_ms).toBe(7000);
  await expect(page.getByRole('alert')).toHaveCount(0);
  const offline = await browser.newContext({ offline: true });
  const report = await offline.newPage();
  const external: string[] = [];
  report.on('request', (req) => {
    if (/^https?:/.test(req.url())) external.push(req.url());
  });
  await report.goto(pathToFileURL(`${folder}/report.html`).href);
  await expect(
    report.getByRole('heading', { name: 'Show the history of Red toolkit' }),
  ).toBeVisible();
  await expect(report.locator('.answer-text')).toContainText(
    'position at the requested time is not confirmed',
  );
  expect(
    await report
      .locator('.frame img')
      .evaluateAll((images) =>
        images.every((image) => (image as HTMLImageElement).naturalWidth > 0),
      ),
  ).toBe(true);
  await expect(report.locator('.box').first()).toBeVisible();
  await report.getByLabel('Show object bounds').uncheck();
  await expect(report.locator('.box').first()).not.toBeVisible();
  await report.getByLabel('Show object bounds').check();
  await report.screenshot({ path: 'test-results/offline-evidence-report.png', fullPage: true });
  await report.setViewportSize({ width: 390, height: 844 });
  expect(await report.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
    true,
  );
  expect(external).toEqual([]);
  await offline.close();
});

test('ambiguous answers offer explicit matches at the original answer cutoff', async ({
  page,
  request,
}) => {
  const video = await (await request.post('/api/v1/demo')).json();
  const run = await (
    await request.post(`/api/v1/videos/${video.id}/runs`, { data: { backend: 'reference' } })
  ).json();
  await expect
    .poll(async () => (await (await request.get(`/api/v1/runs/${run.id}`)).json()).status, {
      timeout: 30_000,
    })
    .toBe('complete');
  await request.post(`/api/v1/runs/${run.id}/questions`, {
    data: { text: 'Where are Red toolkit and Blue remote?', at_ms: 7000, use_provider: false },
  });
  await page.goto('/');
  const choices = page.getByLabel('Choose a matching object');
  await expect(choices.getByRole('button')).toHaveCount(2);
  await expect(page.locator('.answer-text')).toContainText('Several registered objects match');
  await expect(page.getByText('No supported location', { exact: true })).toBeVisible();
  const cutoff = page.getByRole('slider', { name: 'Query cutoff' });
  await cutoff.fill('11000');
  await cutoff.dispatchEvent('change');
  await expect
    .poll(() => page.locator('video').evaluate((video: HTMLVideoElement) => !video.seeking))
    .toBe(true);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/object-clarification.png', fullPage: true });
  const outgoing = page.waitForRequest(
    (req) => req.method() === 'POST' && req.url().endsWith('/questions'),
  );
  await choices.getByRole('button', { name: 'Red toolkit' }).click();
  const payload = (await outgoing).postDataJSON();
  expect(payload.at_ms).toBe(7000);
  expect(payload.object_id).toBe(
    video.objects.find((obj: { name: string }) => obj.name === 'Red toolkit').id,
  );
  await expect(page.locator('.answer-text').filter({ hasText: 'not confirmed' })).toBeVisible();
  await expect(page.getByRole('combobox', { name: 'Query scope' })).toHaveValue(payload.object_id);
  await page.reload();
  await expect(page.locator('.answer-text').filter({ hasText: 'not confirmed' })).toBeVisible();
});

test('suggested objects prefill registration and never register by themselves', async ({
  page,
  request,
}) => {
  const sample = await (await request.post('/api/v1/demo')).json();
  const media = await request.get(sample.media_url);
  // The suggestion button only appears when the server advertises a discovery backend, and
  // the proposals themselves come from a model. Both are stubbed so this checks the browser
  // contract — prefill, no silent registration — rather than model accuracy.
  await page.route('**/api/v1/capabilities', async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    await route.fulfill({ json: { ...body, discovery: { rtdetr: true, cosmos: false } } });
  });
  await page.route('**/api/v1/videos/*/suggestions', (route) =>
    route.fulfill({
      json: {
        at_ms: 0,
        backend: 'rtdetr',
        elapsed_seconds: 0.4,
        limits: 'Proposals describe this one frame only.',
        provenance: { adapter: 'rtdetr-proposals', device: 'mock' },
        proposals: [
          {
            // A COCO category outside the dialog's common list must still be selectable.
            suggested_name: 'Laptop',
            label: 'laptop',
            box: { x1: 0.53125, y1: 0.65278, x2: 0.67188, y2: 0.83056 },
            score: 0.91,
          },
        ],
      },
    }),
  );
  await page.goto('/');
  await page.getByLabel('Upload video file').setInputFiles({
    name: 'Suggested desk.mp4',
    mimeType: 'video/mp4',
    buffer: await media.body(),
  });
  await expect(page.getByRole('heading', { name: 'Suggested desk', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Register object', exact: true }).click();
  const modal = page.getByRole('dialog', { name: 'Register an object' });
  await modal.getByRole('button', { name: /Suggest objects \(RT-DETR\)/ }).click();

  const proposal = modal.getByRole('button', { name: 'Use suggestion Laptop' });
  await expect(proposal).toBeVisible();
  await expect(modal.getByText(/suggestions\s+are not evidence/)).toBeVisible();
  // A proposal on screen must not have registered anything yet.
  await expect(page.locator('.object-chip').filter({ hasText: 'Laptop' })).toHaveCount(0);

  await proposal.click();
  await expect(modal.getByRole('textbox', { name: 'Object name' })).toHaveValue('Laptop');
  await expect(modal.getByRole('combobox', { name: 'Detection category' })).toHaveValue('laptop');
  // An accepted proposal is a starting point, not a commitment: the box stays editable.
  await modal.getByText('Adjust normalized bounds with keyboard').click();
  await expect(modal.getByRole('spinbutton', { name: 'Box left' })).toHaveValue('0.53125');

  await modal.getByRole('button', { name: 'Save reference' }).click();
  await expect(modal).not.toBeVisible();
  await expect(page.locator('.object-chip').filter({ hasText: 'Laptop' })).toBeVisible();
});

test('a knocked camera keeps the region the user drew, and says so', async ({ page, request }) => {
  for (const video of await (await request.get('/api/v1/videos')).json()) {
    await request.delete(`/api/v1/videos/${video.id}`);
  }
  const browserErrors: string[] = [];
  page.on('pageerror', (error) => browserErrors.push(error.message));
  await page.goto('/');
  await page.getByRole('button', { name: 'Try bumped camera' }).click();
  await expect(page.getByText('120 observations', { exact: false })).toBeVisible({
    timeout: 30_000,
  });
  const cutoff = page.getByRole('slider', { name: 'Query cutoff' });
  await cutoff.fill('11000');
  await cutoff.dispatchEvent('change');
  // The camera was knocked at six seconds and nothing on the desk moved afterwards. The remote
  // now occupies the left third of the picture; the place the user drew is still the centre.
  await page
    .getByRole('textbox', { name: 'Ask your memory', exact: true })
    .fill('Where is Blue remote?');
  await page.getByRole('button', { name: 'Send question' }).click();
  await expect(page.locator('.answer-text')).toContainText('Center area');
  await expect(page.locator('.answer-text')).toContainText('camera had moved');
  await expect(page.locator('.evidence-camera').first()).toBeVisible();
  // The regions the user drew are projected into the moved frame, so the claim is visible and
  // not only asserted: three registered areas, with the named one highlighted.
  await expect(page.locator('.evidence-regions polygon')).toHaveCount(3);
  await expect(page.locator('.evidence-regions polygon.named')).toHaveCount(1);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/bumped-camera.png', fullPage: true });
  expect(browserErrors).toEqual([]);
});
