import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const pluginRoot = path.resolve(__dirname, '../..');
const repoRoot = path.resolve(__dirname, '../../../..');

function readPluginFile(relativePath) {
  return fs.readFileSync(path.join(pluginRoot, relativePath), 'utf8');
}

function pluginFileExists(relativePath) {
  return fs.existsSync(path.join(pluginRoot, relativePath));
}

function collectSourceFiles(root) {
  const result = [];
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const fullPath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      result.push(...collectSourceFiles(fullPath));
    } else if (/\.(ts|tsx)$/.test(entry.name)) {
      result.push(fullPath);
    }
  }
  return result;
}

test('declares the splatting viewport host and shader permission', () => {
  const manifest = readPluginFile('witwin.toml');

  assert.match(manifest, /id = "witwin\.splatting"/);
  assert.match(manifest, /entry = "viewport\.js"/);
  assert.match(manifest, /sdkVersion = "1\.0\.0"/);
  assert.match(manifest, /frontend = true/);
  assert.match(manifest, /shader = true/);
});

test('builds a TypeScript-first viewport bundle with Spark and GPU 2DGS code', () => {
  assert.equal(pluginFileExists('web/viewport.js'), true);
  assert.equal(pluginFileExists('web/src/viewport.ts'), true);
  assert.equal(pluginFileExists('web/src/viewport.js'), false);

  const bundle = readPluginFile('web/viewport.js');

  assert.match(bundle, /SplatMesh/);
  assert.match(bundle, /Spark/);
  assert.match(bundle, /GpuBitonicSort/);
  assert.match(bundle, /witwin\.splatting\.gaussianSplatRenderer/);
  assert.match(bundle, /quad2dgsVertexShader/);
  assert.match(bundle, /var REVISION = hostThree\.REVISION/);
  assert.match(bundle, /var threeRevision = Number\.parseInt\(REVISION\)/);
  assert.doesNotMatch(bundle, /Number\.parseInt\(void 0\)/);
  assert.doesNotMatch(bundle, /react\/jsx-runtime/);
  assert.doesNotMatch(bundle, /__witwin\.splatting/);
  assert.doesNotMatch(bundle, /__witwin\.rendering/);
  assert.doesNotMatch(bundle, /node_modules\/three/);
});

test('keeps 2DGS render path free of CPU ordering calls', () => {
  const roots = ['web/src/gizmos', 'web/src/gpu'];
  const files = roots.flatMap((root) => collectSourceFiles(path.join(pluginRoot, root)));
  const source = files.map((file) => fs.readFileSync(file, 'utf8')).join('\n');

  assert.doesNotMatch(source, /\.sort\s*\(/);
  assert.doesNotMatch(source, /Array\.prototype\.sort/);
  assert.doesNotMatch(source, /TypedArray\.prototype\.sort/);
});

test('keeps 2DGS exact sorting and weighted mode on explicit GPU paths', () => {
  const sortShader = readPluginFile('web/src/shaders/gpuSortShaders.ts');
  const sortPass = readPluginFile('web/src/gpu/GpuBitonicSort.ts');
  const dataTextures = readPluginFile('web/src/gpu/SplatDataTextures.ts');
  const quadVisualizer = readPluginFile('web/src/gizmos/Quad2DGSVisualizer.tsx');
  const loader = readPluginFile('web/src/data/load2dgs.ts');

  assert.match(sortShader, /pairIsLowerIndex/);
  assert.match(sortShader, /takePartnerForLower/);
  assert.match(sortShader, /takePartnerForUpper/);
  assert.match(sortPass, /sortTexelCount/);
  assert.match(dataTextures, /sortTexelCount/);
  assert.match(quadVisualizer, /buildWeightedBlendMaterial/);
  assert.match(quadVisualizer, /THREE\.CustomBlending/);
  assert.match(loader, /const payloadOffset = align4\(attrsEnd\)/);
  assert.match(loader, /let offset = payloadOffset/);
});

test('uses complete nearest-filtered float render targets for 2DGS GPU textures', () => {
  const depthPass = readPluginFile('web/src/gpu/DepthKeyPass.ts');
  const sortPass = readPluginFile('web/src/gpu/GpuBitonicSort.ts');
  const passState = readPluginFile('web/src/gpu/renderPassState.ts');

  for (const source of [depthPass, sortPass]) {
    assert.match(source, /minFilter: THREE\.NearestFilter/);
    assert.match(source, /magFilter: THREE\.NearestFilter/);
    assert.match(source, /generateMipmaps: false/);
    assert.match(source, /beginFullscreenTargetPass/);
    assert.match(source, /finishFullscreenTargetPass/);
  }
  assert.match(passState, /renderer\.getViewport/);
  assert.match(passState, /renderer\.getScissor/);
  assert.match(passState, /renderer\.getScissorTest/);
  assert.match(passState, /renderer\.autoClear/);
  assert.match(passState, /renderer\.xr\.enabled/);
});

test('renders 2DGS quads double-sided so surface splats remain visible when orbiting', () => {
  assert.match(readPluginFile('web/src/gizmos/Quad2DGSVisualizer.tsx'), /side: THREE\.DoubleSide/);
});

test('uses canonical Three position vertices for 2DGS instanced quads', () => {
  const quadVisualizer = readPluginFile('web/src/gizmos/Quad2DGSVisualizer.tsx');
  const quadShaders = readPluginFile('web/src/shaders/quad2dgsShaders.ts');

  assert.match(quadVisualizer, /geometry\.setAttribute\(\s*'position'/);
  assert.doesNotMatch(quadVisualizer, /geometry\.setAttribute\('corner'/);
  assert.match(quadShaders, /in vec3 position;/);
  assert.match(quadShaders, /vec2 localUv = position\.xy;/);
  assert.doesNotMatch(quadShaders, /in vec2 corner;/);
});

test('does not duplicate GLSL version directives when RawShaderMaterial uses GLSL3', () => {
  const gpuSortShaders = readPluginFile('web/src/shaders/gpuSortShaders.ts');
  const quadShaders = readPluginFile('web/src/shaders/quad2dgsShaders.ts');
  const depthPass = readPluginFile('web/src/gpu/DepthKeyPass.ts');
  const sortPass = readPluginFile('web/src/gpu/GpuBitonicSort.ts');
  const quadVisualizer = readPluginFile('web/src/gizmos/Quad2DGSVisualizer.tsx');
  const bundle = readPluginFile('web/viewport.js');

  assert.match(depthPass, /glslVersion: THREE\.GLSL3/);
  assert.match(sortPass, /glslVersion: THREE\.GLSL3/);
  assert.match(quadVisualizer, /glslVersion: THREE\.GLSL3/);
  assert.doesNotMatch(gpuSortShaders, /#version 300 es/);
  assert.doesNotMatch(quadShaders, /#version 300 es/);
  assert.doesNotMatch(bundle, /var depthKeyVertexShader = `#version 300 es/);
  assert.doesNotMatch(bundle, /var depthKeyFragmentShader = `#version 300 es/);
  assert.doesNotMatch(bundle, /var bitonicSortFragmentShader = `#version 300 es/);
  assert.doesNotMatch(bundle, /var quad2dgsVertexShader = `#version 300 es/);
  assert.doesNotMatch(bundle, /var quad2dgsFragmentShader = `#version 300 es/);
});

test('disables raycast on splat renderer meshes so viewport picking cannot crash on custom geometry', () => {
  assert.match(readPluginFile('web/src/gizmos/utils.ts'), /export function disableRaycast/);
  assert.match(readPluginFile('web/src/gizmos/utils.ts'), /child\.raycast = \(\) => undefined/);
  assert.match(readPluginFile('web/src/gizmos/SparkSplatVisualizer.tsx'), /disableRaycast\(activeMesh/);
  assert.match(readPluginFile('web/src/gizmos/Quad2DGSVisualizer.tsx'), /disableRaycast\(mesh\)/);
});

test('matches empty GaussianSplatRenderer components so in-component Generate can activate the visualizer', () => {
  assert.match(readPluginFile('web/src/data/extractSplatComponent.ts'), /export function hasSplatRendererComponent/);
  assert.match(readPluginFile('web/src/gizmos/SplattingVisualizer.tsx'), /return hasSplatRendererComponent\(object\);/);
  assert.doesNotMatch(readPluginFile('web/src/gizmos/SplattingVisualizer.tsx'), /return !!findSplatComponent\(object\);/);
});

test('installs SparkRenderer into the R3F scene for 3DGS SplatMesh rendering', () => {
  const sparkVisualizer = readPluginFile('web/src/gizmos/SparkSplatVisualizer.tsx');

  assert.match(sparkVisualizer, /SparkRenderer/);
  assert.match(sparkVisualizer, /function SparkRendererBridge/);
  assert.match(sparkVisualizer, /scene\.add\(record\.renderer\)/);
  assert.match(sparkVisualizer, /scene\.remove\(record\.renderer\)/);
  assert.match(sparkVisualizer, /React\.createElement\(SparkRendererBridge/);
});

test('does not rely on old Studio Gaussian splat SDK helpers', () => {
  assert.equal(fs.existsSync(path.join(repoRoot, 'studio/src/features/viewport/host/types.ts')), true);
  assert.doesNotMatch(fs.readFileSync(path.join(repoRoot, 'studio/src/features/viewport/host/types.ts'), 'utf8'), /GaussianSplatResource/);
  assert.doesNotMatch(fs.readFileSync(path.join(repoRoot, 'studio/packages/witwin-viewport/src/index.ts'), 'utf8'), /loadGaussianSplats/);
  assert.doesNotMatch(fs.readFileSync(path.join(repoRoot, 'studio/src/features/viewport/host/assetApi.ts'), 'utf8'), /loadGaussianSplats/);
});
