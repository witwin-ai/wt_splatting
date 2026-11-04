import type { FrontendHost } from '@witwin/host';
import SplattingVisualizer, { matchesSplattingObject } from './gizmos/SplattingVisualizer';
import PointCloudVisualizer, { pointCloudVisualizerConfig } from './gizmos/PointCloudVisualizer';
import { React } from './runtime';

function findComponent(object: any, predicate: (component: any) => boolean): any | null {
  const components = Array.isArray(object.components) ? object.components : [];
  return components.find(predicate) || null;
}

let disposables: Array<{ dispose(): void }> = [];

export function activateSplattingViewport(host: FrontendHost): void {
  disposables.push(
    host.registerVisualizer({
      id: 'witwin.splatting.gaussianSplatRenderer',
      priority: 95,
      skipStandardGeometry: true,
      match: matchesSplattingObject,
      component: (props) => React.createElement(SplattingVisualizer, { ...props, assets: host.assets }),
    }),
    host.registerVisualizer({
      id: 'witwin.splatting.pointcloud',
      priority: 90,
      skipStandardGeometry: true,
      match: (object) => !!findComponent(object, pointCloudVisualizerConfig.matcher),
      component: ({ object, selected, hovered }) => {
        const component = findComponent(object, pointCloudVisualizerConfig.matcher);
        const data = component ? pointCloudVisualizerConfig.extractor(component, object) : null;
        return data
          ? React.createElement(PointCloudVisualizer, {
              componentData: data,
              assets: host.assets,
              selected,
              hovered,
            })
          : null;
      },
    })
  );
}

export function deactivateSplattingViewport(): void {
  for (const disposable of disposables) {
    disposable.dispose();
  }
  disposables = [];
}
