import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { licenseNotices } from './license-notices.js';
export default defineConfig({
  plugins: [svelte(), licenseNotices()],
  server: { proxy: { '/api': 'http://127.0.0.1:9000' } },
});
