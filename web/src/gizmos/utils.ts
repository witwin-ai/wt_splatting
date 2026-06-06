import { React, Text } from '../runtime';
import type * as ThreeTypes from 'three';

export function Marker({ message, color = '#ffb020' }: { message: string; color?: string }) {
  return React.createElement(
    Text as any,
    { fontSize: 0.18, color, anchorX: 'center', anchorY: 'middle', position: [0, 0.35, 0] },
    message
  );
}

export function hasReadableSparkSuffix(url: string): boolean {
  return /\.(ply|spz|splat|ksplat|sog|zip|rad)(?:[?#].*)?$/i.test(url);
}

export function disableRaycast(object: ThreeTypes.Object3D): void {
  object.traverse((child) => {
    child.raycast = () => undefined;
  });
}
