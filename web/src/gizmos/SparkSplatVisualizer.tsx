import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import { React, THREE, useEffect, useState, useThree } from '../runtime';
import type * as ThreeTypes from 'three';
import type { ViewportAssetsApi } from '../sdk/types';
import type { SplatRendererData } from '../data/extractSplatComponent';
import { Marker, disableRaycast, hasReadableSparkSuffix } from './utils';

type SparkRendererRecord = {
  renderer: SparkRenderer;
  refs: number;
};

const sparkRenderers = new WeakMap<ThreeTypes.WebGLRenderer, SparkRendererRecord>();

export default function SparkSplatVisualizer({ data, assets }: { data: SplatRendererData; assets: ViewportAssetsApi }) {
  const [mesh, setMesh] = useState<SplatMesh | null>(null);
  const [error, setError] = useState<string | null>(data.error || null);

  useEffect(() => {
    let cancelled = false;
    let activeMesh: SplatMesh | null = null;
    setError(data.error || null);
    setMesh(null);

    async function load(): Promise<void> {
      try {
        const url = assets.resolve(data.ref);
        if (!hasReadableSparkSuffix(url)) {
          throw new Error('Spark splat URL must preserve .ply/.spz/.splat/.ksplat/.sog/.zip/.rad suffix.');
        }
        activeMesh = new SplatMesh({ url, maxSplats: data.maxSplats });
        activeMesh.opacity = data.opacityScale;
        activeMesh.scale.setScalar(data.scaleModifier);
        disableRaycast(activeMesh as unknown as ThreeTypes.Object3D);
        if ('maxSh' in activeMesh) activeMesh.maxSh = data.shDegree;
        await activeMesh.initialized;
        if (!cancelled) setMesh(activeMesh);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      }
    }

    load();
    return () => {
      cancelled = true;
      activeMesh?.dispose();
      setMesh(null);
    };
  }, [assets, data.ref.assetId, data.ref.contentHash, data.format, data.maxSplats, data.opacityScale, data.scaleModifier, data.shDegree, data.error]);

  if (error) return React.createElement(Marker, { message: error, color: '#ff5a5f' });
  if (!mesh) return React.createElement(Marker, { message: 'Loading splats...', color: '#8aa4ff' });
  return React.createElement(
    React.Fragment,
    null,
    React.createElement(SparkRendererBridge),
    React.createElement('primitive' as any, { object: mesh as unknown as ThreeTypes.Object3D })
  );
}

function SparkRendererBridge() {
  const { gl, scene, invalidate } = useThree();

  useEffect(() => {
    let rec = sparkRenderers.get(gl);
    if (!rec) {
      rec = {
        renderer: new SparkRenderer({ renderer: gl, onDirty: invalidate }),
        refs: 0,
      };
      disableRaycast(rec.renderer as unknown as ThreeTypes.Object3D);
      sparkRenderers.set(gl, rec);
    }
    const record = rec;
    record.refs += 1;
    if (record.renderer.parent !== scene) {
      record.renderer.parent?.remove(record.renderer);
      scene.add(record.renderer);
    }

    return () => {
      record.refs -= 1;
      if (record.refs <= 0) {
        scene.remove(record.renderer);
        record.renderer.dispose();
        sparkRenderers.delete(gl);
      }
    };
  }, [gl, scene, invalidate]);

  return null;
}
