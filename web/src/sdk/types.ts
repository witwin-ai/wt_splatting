import type { ComponentType, ReactNode } from 'react';

export interface Disposable {
  dispose(): void;
}

export type OverlayZone =
  | 'toolbar.left'
  | 'toolbar.right'
  | 'toolbar.bottom'
  | 'overlay.topLeft'
  | 'overlay.topRight'
  | 'overlay.bottomLeft'
  | 'overlay.bottomRight';

export interface HostSceneObject {
  id: string;
  name: string;
  components?: unknown[];
  [key: string]: unknown;
}

export interface RenderResourceRef {
  assetId: string;
  uri: string;
  kind: 'volume' | 'sdf' | 'point_cloud' | 'gaussian_splats' | 'gaussian_splats_2d' | 'mesh' | 'ray_batch' | string;
  format: string;
  byteLength?: number;
  contentHash?: string;
  version?: number;
  sceneId?: string;
  shape?: number[];
  bounds?: [number, number, number, number, number, number] | number[];
  attributes?: Record<string, unknown>;
}

export interface BinaryFetchOptions {
  signal?: AbortSignal;
  onProgress?: (loaded: number, total?: number) => void;
}

export interface ViewportAssetsApi {
  resolve(ref: RenderResourceRef | string): string;
  fetchBinary(ref: RenderResourceRef | string, options?: BinaryFetchOptions): Promise<ArrayBuffer>;
  loadVolume(ref: RenderResourceRef, options?: BinaryFetchOptions): Promise<unknown>;
  loadPointCloud(ref: RenderResourceRef, options?: BinaryFetchOptions): Promise<unknown>;
  release(resource: { dispose?: () => void }): void;
}

export interface VisualizerConfig {
  id: string;
  match: (object: HostSceneObject) => boolean;
  component: ComponentType<{ object: HostSceneObject; selected: boolean; hovered: boolean }>;
  priority?: number;
  skipStandardGeometry?: boolean;
}

export interface ViewportHost {
  readonly version: string;
  readonly pluginId: string;
  registerVisualizer(cfg: VisualizerConfig): Disposable;
  registerGizmo(cfg: {
    id: string;
    match: (object: HostSceneObject) => boolean;
    component: ComponentType<{ object: HostSceneObject; selected: boolean; hovered: boolean }>;
    priority?: number;
  }): Disposable;
  registerOverlay(zone: OverlayZone, render: () => ReactNode): Disposable;
  registerPass(pass: { id: string; component: ComponentType<Record<string, unknown>>; priority?: number }): Disposable;
  registerTool(tool: Record<string, unknown>): Disposable;
  readonly scene: {
    getObjects(): Record<string, HostSceneObject>;
    getSelection(): string[];
    subscribe(listener: () => void): () => void;
  };
  readonly commands: { execute(id: string, args?: unknown): Promise<unknown> };
  readonly contextKeys: { set(k: string, v: unknown): void; get(k: string): unknown };
  readonly assets: ViewportAssetsApi;
  readonly interaction: { setGizmoDragging(active: boolean): void; isGizmoDragging(): boolean };
  readonly sceneWrites: unknown;
}

export interface ViewportHostModule {
  sdkVersion?: string;
  activate(host: ViewportHost): void | Promise<void>;
  deactivate?(): void | Promise<void>;
}
