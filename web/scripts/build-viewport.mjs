import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const here = path.dirname(fileURLToPath(import.meta.url));
const packageRoot = path.resolve(here, '..');
const repoRoot = path.resolve(packageRoot, '..', '..', '..');
const requireFromStudio = createRequire(path.join(repoRoot, 'studio', 'package.json'));
const requireFromPlugin = createRequire(path.join(packageRoot, 'package.json'));
const esbuild = requireFromStudio('esbuild');
const semver = requireFromStudio('semver');
const args = new Set(process.argv.slice(2));
const watch = args.has('--watch');
const analyze = args.has('--analyze');

const threeHost = path.join(packageRoot, 'src/vendor/three-host.ts');
const sparkPkgPath = findPackageJson(requireFromPlugin.resolve('@sparkjsdev/spark'));
const sparkPkg = JSON.parse(fs.readFileSync(sparkPkgPath, 'utf8'));
const studioPkg = JSON.parse(fs.readFileSync(path.join(repoRoot, 'studio', 'package.json'), 'utf8'));

verifySparkThreeCompatibility(studioPkg.dependencies?.three, sparkPkg);
verifyThreeShimCoverage();

const singletonAliases = new Map([
  ['three', threeHost],
  ['three/examples/jsm/postprocessing/Pass.js', path.join(packageRoot, 'src/vendor/three-pass-host.ts')],
  ['three/addons/postprocessing/Pass.js', path.join(packageRoot, 'src/vendor/three-pass-host.ts')],
  ['react', path.join(packageRoot, 'src/vendor/react-host.ts')],
  ['@react-three/fiber', path.join(packageRoot, 'src/vendor/fiber-host.ts')],
  ['@react-three/drei', path.join(packageRoot, 'src/vendor/drei-host.ts')],
]);

const hostAliasPlugin = {
  name: 'host-singleton-alias',
  setup(build) {
    build.onResolve({ filter: /.*/ }, (args) => {
      const target = singletonAliases.get(args.path);
      if (!target) return null;
      return { path: target };
    });
  },
};

const options = {
  entryPoints: [path.join(packageRoot, 'src/viewport.ts')],
  bundle: true,
  format: 'esm',
  platform: 'browser',
  target: 'es2020',
  jsx: 'transform',
  jsxFactory: 'React.createElement',
  jsxFragment: 'React.Fragment',
  tsconfigRaw: {
    compilerOptions: {
      jsx: 'react',
    },
  },
  outfile: path.join(packageRoot, 'viewport.js'),
  sourcemap: false,
  metafile: true,
  legalComments: 'none',
  logLevel: 'info',
  plugins: [hostAliasPlugin],
};

if (watch) {
  const ctx = await esbuild.context(options);
  await ctx.watch();
  console.log('[wt_splatting] watching TypeScript viewport sources');
} else {
  const result = await esbuild.build(options);
  verifyBundleBoundary(result.metafile);
  if (analyze && result.metafile) {
    const output = path.join(packageRoot, 'viewport.js');
    const outputMeta = result.metafile.outputs[output] || result.metafile.outputs[path.relative(process.cwd(), output)];
    const bytes = outputMeta?.bytes ?? 0;
    console.log(`[wt_splatting] viewport bundle: ${(bytes / 1024).toFixed(1)} KiB`);
    console.log(await esbuild.analyzeMetafile(result.metafile, { verbose: false }));
  }
}

function verifySparkThreeCompatibility(hostRange, sparkPackage) {
  const sparkRange = sparkPackage.peerDependencies?.three || sparkPackage.devDependencies?.three || sparkPackage.dependencies?.three;
  if (!sparkRange) {
    console.log(`[wt_splatting] Spark ${sparkPkg.version} does not declare a three peer range; using Studio ${hostRange}`);
    return;
  }
  const hostVersion = semver.minVersion(hostRange);
  if (!hostVersion || !semver.satisfies(hostVersion, sparkRange)) {
    throw new Error(`Spark ${sparkPkg.version} requires three ${sparkRange}, but Studio declares ${hostRange}`);
  }
}

function verifyThreeShimCoverage() {
  const sparkRoot = path.dirname(sparkPkgPath);
  const shim = fs.readFileSync(threeHost, 'utf8');
  const imported = new Set();
  for (const file of walkFiles(sparkRoot)) {
    if (!/\.[cm]?js$/.test(file)) continue;
    const source = fs.readFileSync(file, 'utf8');
    const sourceWithoutStrings = stripStringLikeContent(source);
    const namespaceAliases = new Set();
    for (const match of source.matchAll(/import\s+\*\s+as\s+([A-Za-z_$][\w$]*)\s+from\s+['"]three['"]/g)) {
      namespaceAliases.add(match[1]);
    }
    for (const alias of namespaceAliases) {
      const memberPattern = new RegExp(`\\b${alias}\\.([A-Za-z_$][\\w$]*)`, 'g');
      for (const match of sourceWithoutStrings.matchAll(memberPattern)) {
        imported.add(match[1]);
      }
    }
    for (const match of source.matchAll(/import\s+\{([^}]+)\}\s+from\s+['"]three['"]/g)) {
      for (const rawName of match[1].split(',')) {
        const name = rawName.trim().split(/\s+as\s+/)[0]?.trim();
        if (name) imported.add(name);
      }
    }
  }
  const missing = [...imported].filter((name) => !new RegExp(`export\\s+const\\s+${name}\\b`).test(shim));
  if (missing.length) {
    throw new Error(`three-host.ts is missing Spark imports: ${missing.sort().join(', ')}`);
  }
}

function stripStringLikeContent(source) {
  let result = '';
  for (let i = 0; i < source.length;) {
    const char = source[i];
    if (char === '"' || char === "'" || char === '`') {
      const quote = char;
      result += ' ';
      i += 1;
      let escaped = false;
      while (i < source.length) {
        const current = source[i];
        result += current === '\n' ? '\n' : ' ';
        i += 1;
        if (escaped) {
          escaped = false;
        } else if (current === '\\') {
          escaped = true;
        } else if (current === quote) {
          break;
        }
      }
    } else if (char === '/' && source[i + 1] === '/') {
      while (i < source.length && source[i] !== '\n') {
        result += ' ';
        i += 1;
      }
    } else if (char === '/' && source[i + 1] === '*') {
      result += '  ';
      i += 2;
      while (i < source.length && !(source[i] === '*' && source[i + 1] === '/')) {
        result += source[i] === '\n' ? '\n' : ' ';
        i += 1;
      }
      if (i < source.length) {
        result += '  ';
        i += 2;
      }
    } else {
      result += char;
      i += 1;
    }
  }
  return result;
}

function verifyBundleBoundary(metafile) {
  const inputs = Object.keys(metafile.inputs);
  const forbidden = inputs.filter((input) =>
    /node_modules[\\/]three[\\/](?!examples[\\/]jsm[\\/]postprocessing[\\/]Pass\.js)/.test(input) ||
    /node_modules[\\/]react[\\/]/.test(input) ||
    /node_modules[\\/]@react-three[\\/]fiber[\\/]/.test(input)
  );
  if (forbidden.length) {
    throw new Error(`Viewport bundle includes host singleton modules: ${forbidden.join(', ')}`);
  }
  if (!inputs.some((input) => /node_modules[\\/]@sparkjsdev[\\/]spark[\\/]/.test(input))) {
    throw new Error('Viewport bundle does not contain Spark code');
  }
}

function* walkFiles(root) {
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const fullPath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      yield* walkFiles(fullPath);
    } else {
      yield fullPath;
    }
  }
}

function findPackageJson(entry) {
  let current = path.dirname(entry);
  while (current !== path.dirname(current)) {
    const candidate = path.join(current, 'package.json');
    if (fs.existsSync(candidate)) return candidate;
    current = path.dirname(current);
  }
  throw new Error(`Could not locate package.json for ${entry}`);
}
