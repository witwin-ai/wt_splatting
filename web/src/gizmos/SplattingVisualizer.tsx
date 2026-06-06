import { React } from '../runtime';
import type { HostSceneObject, ViewportAssetsApi } from '../sdk/types';
import { findSplatComponent, hasSplatRendererComponent } from '../data/extractSplatComponent';
import Quad2DGSVisualizer from './Quad2DGSVisualizer';
import SparkSplatVisualizer from './SparkSplatVisualizer';
import { Marker } from './utils';

export function matchesSplattingObject(object: HostSceneObject): boolean {
  return hasSplatRendererComponent(object);
}

export default function SplattingVisualizer({
  object,
  assets,
}: {
  object: HostSceneObject;
  selected: boolean;
  hovered: boolean;
  assets: ViewportAssetsApi;
}) {
  const data = findSplatComponent(object);
  if (!data) return React.createElement(Marker, { message: 'Missing splat data' });
  if (data.splatKind === '3dgs') {
    if (!data.format.startsWith('spark.')) {
      return React.createElement(Marker, { message: `Invalid 3DGS format: ${data.format}`, color: '#ff5a5f' });
    }
    return React.createElement(SparkSplatVisualizer, { data, assets });
  }
  if (data.format !== 'witwin.2dgs.quad.v1') {
    return React.createElement(Marker, { message: `Invalid 2DGS format: ${data.format}`, color: '#ff5a5f' });
  }
  return React.createElement(Quad2DGSVisualizer, { data, assets });
}
