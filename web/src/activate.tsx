import type { ViewportHost } from './sdk/types';
import SplattingVisualizer, { matchesSplattingObject } from './gizmos/SplattingVisualizer';
import { React } from './runtime';

let disposables: Array<{ dispose(): void }> = [];

export function activateSplattingViewport(host: ViewportHost): void {
  disposables.push(
    host.registerVisualizer({
      id: 'witwin.splatting.gaussianSplatRenderer',
      priority: 95,
      skipStandardGeometry: true,
      match: matchesSplattingObject,
      component: (props) => React.createElement(SplattingVisualizer, { ...props, assets: host.assets }),
    })
  );
}

export function deactivateSplattingViewport(): void {
  for (const disposable of disposables) {
    disposable.dispose();
  }
  disposables = [];
}
