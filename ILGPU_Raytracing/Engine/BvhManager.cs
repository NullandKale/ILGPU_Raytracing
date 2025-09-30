
// ==============================
// File: Engine/BvhManager.cs
// Minimal change: forward extended views (signature expanded)
// ==============================
using System;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace ILGPU_Raytracing.Engine
{
    public enum RebuildPolicy
    {
        Auto = 0,
        ForceRefit = 1,
        ForceRebuild = 2
    }

    public sealed class BvhManager : IDisposable
    {
        private readonly CudaAccelerator _cuda;
        private Scene _scene;

        public BvhManager(CudaAccelerator cuda, Scene scene) { _cuda = cuda ?? throw new ArgumentNullException(nameof(cuda)); _scene = scene ?? throw new ArgumentNullException(nameof(scene)); }
        public void AttachScene(Scene scene) { _scene = scene ?? throw new ArgumentNullException(nameof(scene)); }
        public void BuildOrRefit(Scene scene, RebuildPolicy policy) { if (scene == null) { throw new ArgumentNullException(nameof(scene)); } _scene = scene; scene.UploadAll(); }

        public void GetDeviceViews(out ArrayView<TLASNode> tlasNodes, out ArrayView<int> tlasInstanceIndices, out ArrayView<InstanceRecord> instances, out ArrayView<BLASNode> blasNodes, out ArrayView<int> spherePrimIndices, out ArrayView<Sphere> spheres, out ArrayView<int> triPrimIndices, out ArrayView<Float3> meshPositions, out ArrayView<MeshTri> meshTris, out ArrayView<Float2> meshTexcoords, out ArrayView<MeshTriUV> meshTriUVs, out ArrayView<int> triMatIndex, out ArrayView<MaterialRecord> materials, out ArrayView<RGBA32> texels, out ArrayView<TexInfo> texInfos)
        {
            if (_scene == null) { throw new InvalidOperationException("No attached scene."); }
            tlasNodes = _scene.TLASNodesView;
            tlasInstanceIndices = _scene.TLASInstanceIndicesView;
            instances = _scene.InstancesView;
            blasNodes = _scene.BLASNodesView;
            spherePrimIndices = _scene.SpherePrimIndicesView;
            spheres = _scene.SpheresView;
            triPrimIndices = _scene.TriPrimIndicesView;
            meshPositions = _scene.MeshPositionsView;
            meshTris = _scene.MeshTrisView;
            meshTexcoords = _scene.MeshTexcoordsView;
            meshTriUVs = _scene.MeshTriUVsView;
            triMatIndex = _scene.TriMaterialIndexView;
            materials = _scene.MaterialsView;
            texels = _scene.TexelsView;
            texInfos = _scene.TexInfosView;
        }

        public void Dispose() { }
    }
}
