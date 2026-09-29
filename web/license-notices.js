import { readFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';

// Keep complete upstream notices beside the JavaScript, icons and font files we ship.
/** @returns {import('vite').Plugin} */
export function licenseNotices() {
  return {
    name: 'findback-license-notices',
    apply: 'build',
    async generateBundle() {
      const packages = new Set();
      for (const id of this.getModuleIds()) {
        const marker = '/node_modules/';
        const index = id.lastIndexOf(marker);
        if (index < 0) continue;
        const base = id.slice(0, index + marker.length);
        const parts = id.slice(base.length).split('/');
        packages.add(base + parts.slice(0, parts[0].startsWith('@') ? 2 : 1).join('/'));
      }
      const sections = ['FindBack frontend dependency licenses\n'];
      for (const directory of [...packages].sort()) {
        const pkg = JSON.parse(await readFile(join(directory, 'package.json'), 'utf8'));
        const names = (await readdir(directory)).filter((name) =>
          /^(licen[sc]e|copying|notice|ofl)([.-]|$)/i.test(name),
        );
        if (!names.length) this.error(`Missing upstream license text for ${pkg.name}`);
        sections.push(`\n=== ${pkg.name} ${pkg.version} (${pkg.license}) ===\n`);
        for (const name of names.sort()) {
          sections.push(await readFile(join(directory, name), 'utf8'));
        }
      }
      this.emitFile({
        type: 'asset',
        fileName: 'assets/THIRD_PARTY_LICENSES.txt',
        source: sections.join('\n'),
      });
    },
  };
}
