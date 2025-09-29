// File: Engine/SceneManager.cs
using System;
using ILGPU;
using ILGPU.Runtime.Cuda;

namespace ILGPU_Raytracing.Engine
{
    public sealed class SceneManager : IDisposable
    {
        private readonly CudaAccelerator _cuda;
        private readonly BvhManager _bvh;
        private Scene _scene;

        public SceneManager(CudaAccelerator cuda)
        {
            _cuda = cuda ?? throw new ArgumentNullException(nameof(cuda));
            _scene = new Scene(_cuda);
            _bvh = new BvhManager(_cuda, _scene);
        }

        public SceneManager(CudaAccelerator cuda, Scene existingScene)
        {
            _cuda = cuda ?? throw new ArgumentNullException(nameof(cuda));
            _scene = existingScene ?? throw new ArgumentNullException(nameof(existingScene));
            _bvh = new BvhManager(_cuda, _scene);
        }

        public Scene Scene
        {
            get
            {
                return _scene;
            }
        }

        public void BuildDefaultScene()
        {
            _scene.BuildDefaultScene();
        }

        public void LoadObjInstance(string objPath, Affine3x4 objectToWorld, float uniformScale = 1f)
        {
            _scene.LoadObjInstance(objPath, objectToWorld, uniformScale);
        }

        public void Commit(RebuildPolicy policy = RebuildPolicy.Auto)
        {
            _bvh.BuildOrRefit(_scene, policy);
        }

        public void GetDeviceViews(out ArrayView<TLASNode> tlasNodes, out ArrayView<int> tlasInstanceIndices, out ArrayView<InstanceRecord> instances, out ArrayView<BLASNode> blasNodes, out ArrayView<int> spherePrimIndices, out ArrayView<Sphere> spheres, out ArrayView<int> triPrimIndices, out ArrayView<Float3> meshPositions, out ArrayView<MeshTri> meshTris)
        {
            _bvh.GetDeviceViews(out tlasNodes, out tlasInstanceIndices, out instances, out blasNodes, out spherePrimIndices, out spheres, out triPrimIndices, out meshPositions, out meshTris);
        }

        public void ReplaceScene(Scene newScene, bool rebuildImmediately = true, RebuildPolicy policy = RebuildPolicy.Auto)
        {
            if (newScene == null)
            {
                throw new ArgumentNullException(nameof(newScene));
            }
            _scene = newScene;
            _bvh.AttachScene(_scene);
            if (rebuildImmediately)
            {
                _bvh.BuildOrRefit(_scene, policy);
            }
        }

        public void Dispose()
        {
            _scene?.Dispose();
            // _bvh currently holds no unmanaged resources; nothing to dispose here for now.
        }
    }
}
