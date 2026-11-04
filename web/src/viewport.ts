import type { PluginDefinition } from '@witwin/host';
import { activateSplattingViewport, deactivateSplattingViewport } from './activate';

export const sdkVersion = '2.0.0';

const plugin: PluginDefinition = {
  id: 'witwin.splatting',
  activate(host) {
    activateSplattingViewport(host);
  },
  deactivate() {
    deactivateSplattingViewport();
  },
};

export default plugin;
