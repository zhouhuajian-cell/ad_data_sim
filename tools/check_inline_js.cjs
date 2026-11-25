const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)]
  .map(m => m[1]).filter(Boolean);
let errors = 0;
scripts.forEach((source, i) => {
  try { new Function(source); }
  catch (error) { errors++; process.stderr.write(`script ${i}: ${error.message}\n`); }
});
process.stdout.write(`checked ${scripts.length} inline script blocks\n`);
process.exitCode = errors ? 1 : 0;
