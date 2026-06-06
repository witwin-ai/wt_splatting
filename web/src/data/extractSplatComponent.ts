import type { HostSceneObject, RenderResourceRef } from '../sdk/types';

export type SplatKind = '3dgs' | '2dgs';
export type SplatSortMode = 'gpu_depth' | 'gpu_weighted';
export type SplatLod = 'auto' | 'full' | 'off';

export interface SplatRendererData {
  ref: RenderResourceRef;
  splatKind: SplatKind;
  format: string;
  splatCount: number;
  shDegree: number;
  opacityScale: number;
  scaleModifier: number;
  sortMode: SplatSortMode;
  maxSplats: number;
  max2dSplats: number;
  lod: SplatLod;
  debugBounds: boolean;
  error?: string;
}

export function hasSplatRendererComponent(object: HostSceneObject): boolean {
  const components = Array.isArray(object.components) ? object.components : [];
  return components.some((component) => isSplatComponent(component));
}

export function findSplatComponent(object: HostSceneObject): SplatRendererData | null {
  const components = Array.isArray(object.components) ? object.components : [];
  for (const component of components) {
    if (!isSplatComponent(component)) continue;
    const values = (component as any).values || component;
    const ref = values?.splats;
    if (!isResourceRef(ref)) return null;
    const sortMode = normalizeSortMode(values?.sort_mode);
    return {
      ref,
      splatKind: normalizeKind(values?.splat_kind),
      format: String(values?.format || ref.format || ''),
      splatCount: numberValue(values?.splat_count, Number(ref.attributes?.splat_count ?? ref.shape?.[0] ?? 0)),
      shDegree: numberValue(values?.sh_degree, 0),
      opacityScale: numberValue(values?.opacity_scale, 1.0),
      scaleModifier: numberValue(values?.scale_modifier, 1.0),
      sortMode: sortMode.mode,
      maxSplats: numberValue(values?.max_splats, 1_500_000),
      max2dSplats: numberValue(values?.max_2d_splats, 262_144),
      lod: normalizeLod(values?.lod),
      debugBounds: !!values?.debug_bounds,
      error: sortMode.error,
    };
  }
  return null;
}

function isSplatComponent(component: unknown): boolean {
  const anyComponent = component as any;
  return (
    anyComponent?.name === 'GaussianSplatRenderer' ||
    anyComponent?.component_name === 'GaussianSplatRenderer' ||
    anyComponent?.base_classes?.includes?.('GaussianSplatRenderer')
  );
}

function isResourceRef(value: unknown): value is RenderResourceRef {
  const ref = value as RenderResourceRef;
  return !!ref && typeof ref === 'object' && typeof ref.assetId === 'string' && typeof ref.uri === 'string';
}

function normalizeKind(value: unknown): SplatKind {
  return value === '2dgs' ? '2dgs' : '3dgs';
}

function normalizeSortMode(value: unknown): { mode: SplatSortMode; error?: string } {
  if (value === 'gpu_weighted') return { mode: 'gpu_weighted' };
  if (value === 'cpu') return { mode: 'gpu_depth', error: 'CPU sorting is not supported for Gaussian Splatting.' };
  return { mode: 'gpu_depth' };
}

function normalizeLod(value: unknown): SplatLod {
  return value === 'full' || value === 'off' ? value : 'auto';
}

function numberValue(value: unknown, fallback: number): number {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}
