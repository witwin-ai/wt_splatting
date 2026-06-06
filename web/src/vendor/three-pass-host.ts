import { BufferGeometry, Float32BufferAttribute, Mesh, OrthographicCamera, Scene } from './three-host';

export class Pass {
  isPass = true;
  enabled = true;
  needsSwap = true;
  clear = false;
  renderToScreen = false;

  setSize(_width: number, _height: number): void {}

  render(): void {
    throw new Error('THREE.Pass.render() must be implemented by subclasses');
  }

  dispose(): void {}
}

export class FullScreenQuad {
  readonly _mesh: InstanceType<typeof Mesh>;
  private readonly _scene: InstanceType<typeof Scene>;
  private readonly _camera: InstanceType<typeof OrthographicCamera>;

  constructor(material?: unknown) {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new Float32BufferAttribute([-1, 3, 0, -1, -1, 0, 3, -1, 0], 3));
    geometry.setAttribute('uv', new Float32BufferAttribute([0, 2, 0, 0, 2, 0], 2));
    this._mesh = new Mesh(geometry, material as any);
    this._scene = new Scene();
    this._camera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this._scene.add(this._mesh);
  }

  dispose(): void {
    this._mesh.geometry.dispose();
  }

  render(renderer: { render(scene: unknown, camera: unknown): void }): void {
    renderer.render(this._scene, this._camera);
  }

  get material(): unknown {
    return this._mesh.material;
  }

  set material(value: unknown) {
    this._mesh.material = value as any;
  }
}
