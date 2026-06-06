import type { ViewportHost } from './sdk/types';
import { activateSplattingViewport, deactivateSplattingViewport } from './activate';

export const sdkVersion = '1.0.0';

export function activate(host: ViewportHost): void {
  activateSplattingViewport(host);
}

export function deactivate(): void {
  deactivateSplattingViewport();
}
